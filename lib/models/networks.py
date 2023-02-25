from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def get_model(args):
    return AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)