import torch
import torch.nn
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ErnieModel




class Fusion(nn.Module):
    '''
    双特征全局局部交互融合
    '''

    def __init__(self, token_len, feature_dim, r=4):
        super(Fusion, self).__init__()
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
        self.layernorm = nn.LayerNorm(args.bert_feature_dim)
        self.ffn = torch.nn.Linear(args.bert_feature_dim, 3072)
        self.output = torch.nn.Linear(3072, args.bert_feature_dim)
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
        self.ablation = args.ablation_learner
        self.pos_embeddings = torch.nn.Embedding(20, args.bert_feature_dim, padding_idx=0)
        self.dep_embeddings = torch.nn.Embedding(2 * args.max_sequence_len - 1, args.bert_feature_dim, padding_idx=0)
        self.tanh = nn.Tanh()
        self.encoder = nn.ModuleList([ EncoderLayer(args) for _ in range(args.layers_num)])
        self.dropout_output = torch.nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(args.bert_feature_dim)
        # self.yasuo = torch.nn.Linear(args.bert_feature_dim *2, args.bert_feature_dim)

        self.pooler = torch.nn.Linear(args.bert_feature_dim, args.bert_feature_dim)

    def forward(self, pos_ids, dep_ids):
        pos_embeds = self.pos_embeddings(pos_ids)
        pos_embeds = self.layernorm(pos_embeds)
        pos_embeds = self.dropout_output(pos_embeds)
        dep_embeds = self.dep_embeddings(dep_ids)
        dep_embeds = self.layernorm(dep_embeds)
        dep_embeds = self.dropout_output(dep_embeds)
        # pos_dep_embeds = torch.cat([pos_embeds, dep_embeds], dim=2)
        # pos_dep_features = self.yasuo(pos_dep_embeds)
        pos_dep_features = pos_embeds + dep_embeds
        if self.ablation == 1:
            return pos_dep_features
        for layer in self.encoder:
            pos_dep_features = layer(pos_dep_features)
        pos_dep_features = self.pooler(pos_dep_features)
        pos_dep_features = self.tanh(pos_dep_features)
        return pos_dep_features

class MultiInferBert(torch.nn.Module):
    def __init__(self, args):
        super(MultiInferBert, self).__init__()
        self.args = args
        if args.is_ernie:
            self.bert = ErnieModel.from_pretrained(args.bert_model_path,return_dict = False)
        else:
            self.bert = BertModel.from_pretrained(args.bert_model_path, return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
        self.learner = PDEncoder(self.args)

        self.fusion = Fusion(args.max_sequence_len, args.bert_feature_dim)

        self.ablation = args.ablation_fusion
        self.cls_linear = torch.nn.Linear(args.bert_feature_dim*2, args.class_num)
        self.feature_linear = torch.nn.Linear(args.bert_feature_dim*2 + args.class_num*3, args.bert_feature_dim*2)
        self.dropout_output = torch.nn.Dropout(0.1)

    def multi_hops(self, features, mask, k):
        '''generate mask'''
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

    def forward(self, tokens, masks, pos_ids, dep_ids):

        bert_feature, _ = self.bert(tokens, masks)
        bert_feature = self.dropout_output(bert_feature)

        pos_dep_features = self.dropout_output(self.learner(pos_ids,dep_ids))
        if self.ablation == 0:
            bert_feature = self.fusion(bert_feature, pos_dep_features)
        else:
            bert_feature = bert_feature + pos_dep_features

        bert_feature = bert_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        bert_feature_T = bert_feature.transpose(1, 2)
        features = torch.cat([bert_feature, bert_feature_T], dim=3)
        logits = self.multi_hops(features, masks, self.args.nhops)

        return logits[-1]
