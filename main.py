from flask import Flask, jsonify, request,json
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def bert_qa(question, context, pipeline_):
    QA_input = {
        'question': question,
        'context': context
    }
    pipeline_ = pipeline_
    return pipeline_(QA_input)


app = Flask(__name__)


@app.route("/", methods=['POST'])
def dummy_api():
    jsondata = request.get_json()
    print(jsondata)
    ans = bert_qa(jsondata['question'], jsondata['context'], nlp)
    score = ans['score']
    answer = ans['answer']
    print(score)
    print(answer)
    result = {'answer': answer, 'score':score}
    return json.dumps(result), 200


#model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model_name = 'mrm8488/electra-small-finetuned-squadv2'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

