"""
Elliot Schumacher, Johns Hopkins University
Created 5/31/19
"""


def main():
    import torch
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

    # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    import logging
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12


if __name__ == "__main__":
    main()