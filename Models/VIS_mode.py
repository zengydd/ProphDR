    def pred(self, pt_path, drug_id, smiles, omic_f, cosmic_id = torch.tensor(0)):
        self.model.load_state_dict(torch.load(pt_path, map_location='cpu')['model_state_dict'])
        encode_D_pred, valid_lens = encoder_D_pred(smiles)
        score = self.model(drug_id, encode_D_pred, omic_f, valid_lens, cosmic_id)
        score = score.flatten().to(self.device)
        print('drug_id', drug_id)
        print('cosmic_id', cosmic_id)
        print('score', score)
        return score
    
    def VIS(self, test_set, pt_path):
        # 在测试集上重新初始化一次
        # global attn_G_dict, attn_D_dict
        # attn_G_dict = nested_dict_factory()
        # attn_D_dict = nested_dict_factory()

        self.model = self.model.to(self.device)
        label = self.label
        params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 0,
                  'drop_last': False}
        # loader
        test_generator = data.DataLoader(
            mydata(
            test_set.index.values,
            test_set[label].values, 
            self.res_df, 
            drug_smiles_df, 
            self.omic_encode_dict
            ), 
            **params)       
        print("=====testing...")
        self.model.load_state_dict(torch.load(pt_path, map_location='cpu')['model_state_dict'])
        metric_result, loss = self.validate(test_generator, self.model)
        return metric_result, loss
    