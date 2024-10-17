from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class data_encoder(Dataset):
    def __init__(self, jsondata, tokenizer):
        self.tokenizer = tokenizer
        self.data = jsondata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        Human_text = data['Human_text']
        AI_text = data['AI_text']
        Rewrite_text = data['Rewrite_text']
        Humanlike_text = data['Humanlike_text']

        encoded_input_1 = self.tokenizer(
            Human_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        input_ids_1 = encoded_input_1.input_ids.squeeze()
        attention_mask_1 = encoded_input_1.attention_mask.squeeze()
        token_type_ids_1 = encoded_input_1.token_type_ids.squeeze()


        encoded_input_2 = self.tokenizer(
            AI_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        input_ids_2 = encoded_input_2.input_ids.squeeze()
        attention_mask_2 = encoded_input_2.attention_mask.squeeze()
        token_type_ids_2 = encoded_input_2.token_type_ids.squeeze()


        encoded_input_3 = self.tokenizer(
            Rewrite_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        input_ids_3 = encoded_input_3.input_ids.squeeze()
        attention_mask_3 = encoded_input_3.attention_mask.squeeze()
        token_type_ids_3 = encoded_input_3.token_type_ids.squeeze()


        encoded_input_4 = self.tokenizer(
            Humanlike_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        input_ids_4 = encoded_input_4.input_ids.squeeze()
        attention_mask_4 = encoded_input_4.attention_mask.squeeze()
        token_type_ids_4 = encoded_input_4.token_type_ids.squeeze()


        return {
            'input_ids_1': input_ids_1,
            'attention_mask_1': attention_mask_1,
            'token_type_ids_1': token_type_ids_1,
            'input_ids_2': input_ids_2,
            'attention_mask_2': attention_mask_2,
            'token_type_ids_2': token_type_ids_2,
            'input_ids_3': input_ids_3,
            'attention_mask_3': attention_mask_3,
            'token_type_ids_3': token_type_ids_3,
            'input_ids_4': input_ids_4,
            'attention_mask_4': attention_mask_4,
            'token_type_ids_4': token_type_ids_4,
        }


############  封装预处理后的数据集
def dataset_loader(dataset, batch_size, shuffle=False):
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader



