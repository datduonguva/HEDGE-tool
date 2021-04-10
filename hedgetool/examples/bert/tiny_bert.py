import tokenization
import modeling


tokenizer = tokenization.FullTokenizer(
    vocab_file="/home/datduong/Downloads/bert_tiny/vocab.txt",
    do_lower_case=False
)

my_text = 'this is a good song for anyone that want to take a look'
print(tokenizer.tokenize(my_text))

print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(my_text)))


# get the config
config = modeling.BertConfig.from_json_file("/home/datduong/Downloads/bert_tiny/bert_config.json")


# create the model

model = modeling.create_model(
    config,
    is_training=True,

