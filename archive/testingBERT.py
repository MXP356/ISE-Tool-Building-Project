from transformers import BertTokenizer,BertModel

print('Hello')


# https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertTokenizer
# which extends https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
## * check parms for the __call__ method
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print('Tokenizer Details:')
print(tokenizer)


#print('\n\n\nTokens:')
#print(tokenizer.tokenize('Hello my name is Muneeb'))

print('\n\n\nTest Embedding:')
inputsForModel = tokenizer('Hello my name is Muneeb', return_tensors='pt', padding=True, truncation=True)
print(inputsForModel)


#print('\n\nDecoded:')
#print(tokenizer.decode(inputsForModel.input_ids))



# https://huggingface.co/docs/transformers/en/model_doc/bert
model = BertModel.from_pretrained('bert-base-uncased')

modelOutput = model(**inputsForModel) # (** unpacks the dictionary)

print('\n\nOUTPUT:',modelOutput)

print('\n\nOutput Keys:', modelOutput.keys())
## We get a 'last_hidden_state' and 'pooler_output' key
# 'pooler_output' is a sentence level embedding which we use for classification tasks
# Nvm: after further research, pooler_output is the CLS Token but processed more (processed through a linear layer and with "tanh activation"),
# using the unprocessed CLS token is better. This can be obtained via: last_hidden_state[:, 0, :]
# this extracts the CLS token from 'last_hidden_state'
# ('last_hidden_state' is a 3D tensor which contains all the token embeddings. 'pooler_output' is a post-processed version of the CLS token.)

embedding = modelOutput.last_hidden_state[:,0,:]
print('EMBEDDING:',embedding) #print('EMBEDDING:',embedding.shape)




multiOutput = tokenizer(['hello','hi'], return_tensors='pt')
print('\n\nMulti Output:',multiOutput) # Ok it works, input_ids, token_type_ids and attention_mask are basically an array of arrays.



print('\n\n\nDone')

