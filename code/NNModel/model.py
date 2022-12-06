import torch
import torch.nn
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from attention_module import MultiHeadedAttention, SelfAttention



class Fusion(nn.Module):
    '''
    双特征全局局部交互融合
    '''

    def __init__(self, token_len, feature_dim,device, r=4):
        super(Fusion, self).__init__()
        self.token_len = token_len
        self.device = device
        inter_dim = feature_dim // r
        inter_len = token_len // r

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Linear(feature_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.GELU(),
            nn.Linear(inter_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Sigmoid()
        )
        # 全局注意力
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.global_att = nn.Sequential(
            nn.Linear(token_len, inter_len),
            nn.LayerNorm(inter_len),
            nn.GELU(),
            nn.Linear(inter_len, token_len),
            nn.LayerNorm(token_len),
            nn.Sigmoid()
        )
        # 第二次
        # 本地注意力
        self.local_att2 = nn.Sequential(
            nn.Linear(feature_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.GELU(),
            nn.Linear(inter_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Sigmoid()
        )
        # 全局注意力
        self.global_att2 = nn.Sequential(
            nn.Linear(token_len, inter_len),
            nn.LayerNorm(inter_len),
            nn.GELU(),
            nn.Linear(inter_len, token_len),
            nn.LayerNorm(token_len),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # print(x.device,self.device)
        x = torch.cat([x,torch.zeros(x.shape[0],self.token_len-x.shape[1],x.shape[2]).to(self.device)],dim=1)
        y = torch.cat([y,torch.zeros(y.shape[0],self.token_len-y.shape[1],y.shape[2]).to(self.device)],dim=1)
        # print(x.shape,y.shape)
        xa = x + y
        l = self.local_att(xa)
        xl_att = l
        yl_att = 1 - l
        g = self.global_att(self.pooler(xa).permute(0,2,1)).permute(0,2,1)
        xx = x * xl_att
        yy = y * yl_att

        z = g * xx + (1 - g) * yy

        l2 = self.local_att2(z)
        xl_att2 = l2
        yl_att2 = 1 - l2
        g2 = self.global_att2(self.pooler(z).permute(0, 2, 1)).permute(0, 2, 1)
        xx2 = xx * xl_att2
        yy2 = yy * yl_att2

        z2 = g2 * xx2 + (1 - g2) * yy2
        return z2


class EncoderLayer(nn.Module):
    def __init__(self,args):
        super(EncoderLayer, self).__init__()
        self.gelu = nn.GELU()
        self.dropout_output = torch.nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(args.lstm_dim*2)
        self.ffn = torch.nn.Linear(args.lstm_dim*2, 3072)
        self.output = torch.nn.Linear(3072, args.lstm_dim*2)
    def forward(self,pos_dep_features):
        pos_dep_features = self.ffn(pos_dep_features)
        pos_dep_features = self.gelu(pos_dep_features)

        pos_dep_features = self.output(pos_dep_features)
        pos_dep_features = self.layernorm(pos_dep_features)
        pos_dep_features = self.dropout_output(pos_dep_features)
        return pos_dep_features

class PDEncoder(nn.Module):
    def __init__(self,args):
        super(PDEncoder, self).__init__()
        self.pos_embeddings = torch.nn.Embedding(20, args.lstm_dim*2, padding_idx=0)
        self.dep_embeddings = torch.nn.Embedding(2 * args.max_sequence_len - 1, args.lstm_dim*2, padding_idx=0)
        self.tanh = nn.Tanh()
        self.encoder = nn.ModuleList([ EncoderLayer(args) for _ in range(args.layers_num)])
        self.dropout_output = torch.nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(args.lstm_dim*2)
        # self.yasuo = torch.nn.Linear(args.bert_feature_dim *2, args.bert_feature_dim)

        self.pooler = torch.nn.Linear(args.lstm_dim*2, args.lstm_dim*2)

    def forward(self, pos_ids, dep_ids,lengths):
        pos_embeds = self.pos_embeddings(pos_ids)
        pos_embeds = self.layernorm(pos_embeds)
        pos_embeds = self.dropout_output(pos_embeds)
        dep_embeds = self.dep_embeddings(dep_ids)
        dep_embeds = self.layernorm(dep_embeds)
        dep_embeds = self.dropout_output(dep_embeds)
        # pos_dep_embeds = torch.cat([pos_embeds, dep_embeds], dim=2)
        # pos_dep_features = self.yasuo(pos_dep_embeds)
        pos_dep_features = pos_embeds + dep_embeds
        pos_dep_features = pack_padded_sequence(pos_dep_features, lengths, batch_first=True)
        pos_dep_features,_ = pad_packed_sequence(pos_dep_features, batch_first=True)
        # print(pos_dep_features.shape)
        for layer in self.encoder:
            pos_dep_features = layer(pos_dep_features)
        pos_dep_features = self.pooler(pos_dep_features)
        pos_dep_features = self.tanh(pos_dep_features)
        # print(pos_dep_features.shape)
        return pos_dep_features


class MultiInferRNNModel(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, args):
        '''double embedding + lstm encoder + dot self attention'''
        super(MultiInferRNNModel, self).__init__()

        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0)

        self.bilstm = torch.nn.LSTM(300+100, args.lstm_dim,
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.attention_layer = SelfAttention(args)

        self.syntax_learner = PDEncoder(self.args)

        self.fusion = Fusion(args.max_sequence_len,args.lstm_dim*2,device=args.device)


        self.feature_linear = torch.nn.Linear(args.lstm_dim*4 + args.class_num*3, args.lstm_dim*4)
        self.cls_linear = torch.nn.Linear(args.lstm_dim*4, args.class_num)

    def _get_embedding(self, sentence_tokens, mask):
        gen_embed = self.gen_embedding(sentence_tokens)
        domain_embed = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([gen_embed, domain_embed], dim=2)
        embedding = self.dropout1(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths, batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def _cls_logits(self, features):
        # features = self.dropout2(features)
        tags = self.cls_linear(features)
        return tags

    def multi_hops(self, features, lengths, mask, k):
        '''generate mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        '''save all logits'''
        logits_list = []
        logits = self._cls_logits(features)
        logits_list.append(logits)

        for i in range(k):
            #probs = torch.softmax(logits, dim=3)
            probs = logits
            logits = probs * mask

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1,-1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)
            features = self.feature_linear(new_features)
            logits = self._cls_logits(features)
            logits_list.append(logits)
        return logits_list

    def forward(self, sentence_tokens, lengths, mask,pos_ids, dep_ids):
        lengths = lengths.to("cpu")
        embedding = self._get_embedding(sentence_tokens, mask)
        # print(embedding.shape)
        lstm_feature = self._lstm_feature(embedding, lengths)
        # self attention
        lstm_feature_attention = self.attention_layer(lstm_feature, lstm_feature, mask[:,:lengths[0]])
        # print(lstm_feature.shape,lstm_feature_attention.shape)
        #lstm_feature_attention = self.attention_layer.forward_perceptron(lstm_feature, lstm_feature, mask[:, :lengths[0]])
        lstm_feature = lstm_feature + lstm_feature_attention

        pos_dep_features = self.syntax_learner(pos_ids, dep_ids,lengths)
        # print(lstm_feature.shape,pos_dep_features.shape)
        lstm_feature = self.fusion(lstm_feature, pos_dep_features)
        lstm_feature = pack_padded_sequence(lstm_feature, lengths, batch_first=True)
        lstm_feature,_ = pad_packed_sequence(lstm_feature, batch_first=True)
        
        lstm_feature = lstm_feature.unsqueeze(2).expand([-1,-1, lengths[0], -1])
        # print(lstm_feature.shape)
        lstm_feature_T = lstm_feature.transpose(1, 2)
        features = torch.cat([lstm_feature, lstm_feature_T], dim=3)

        logits = self.multi_hops(features, lengths, mask, self.args.nhops)
        return [logits[-1]]


class MultiInferCNNModel(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, args):
        super(MultiInferCNNModel, self).__init__()
        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        self.attention_layer = SelfAttention(args)

        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(0.5)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)

        self.feature_linear = torch.nn.Linear(args.cnn_dim*2 + args.class_num*3, args.cnn_dim*2)
        self.cls_linear = torch.nn.Linear(256*2, args.class_num)

    def multi_hops(self, features, lengths, mask, k):
        '''generate mtraix mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        '''save all logits'''
        logits_list = []
        logits = self.cls_linear(features)
        logits_list.append(logits)

        for i in range(k):
            #probs = torch.softmax(logits, dim=3)
            probs = logits
            logits = probs * mask

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1,-1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)
            features = self.feature_linear(new_features)
            logits = self.cls_linear(features)
            logits_list.append(logits)
        return logits_list

    def forward(self, x, x_len, x_mask):
        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(x)), dim=2)
        x_emb = self.dropout(x_emb).transpose(1, 2)
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x_conv = x_conv[:, :x_len[0], :]

        feature_attention = self.attention_layer.forward_perceptron(x_conv, x_conv, x_mask[:, :x_len[0]])
        x_conv = x_conv + feature_attention

        x_conv = x_conv.unsqueeze(2).expand([-1, -1, x_len[0], -1])
        x_conv_T = x_conv.transpose(1, 2)
        features = torch.cat([x_conv, x_conv_T], dim=3)

        logits = self.multi_hops(features, x_len, x_mask, self.args.nhops)
        return [logits[-1]]

