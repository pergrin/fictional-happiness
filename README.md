# fictional-happiness

Python codes for joint relation and entity extraction.

Used Data:
[Link to sciERC dataset]: https://paperswithcode.com/dataset/scierc
The input raw file as well as formatted files for the publically available sciERC dataset, have been shared herewith.

Users can use there own or other datasets after converting there respective inputs to the same format as of the files in the formatted_data.zip folder.

![Joint Training](https://miro.medium.com/max/3688/1*rrIJOpJO8fkFECNHlwq-jQ.png)


Steps:
1. Format input data to required format as mentioned:
```diff
{
    "mentions": [
        {
            "begin": 2,
            "id": "m0",
            "end": 7,
            "type": "Generic",
            "text": "model"
        }....],
	"relations": [
        {
            "args": [
                "m0",
                "m1"
            ],
            "type": "USED-FOR",
            "id": "r0"
        }....],
	"sentences": [
        {
            "begin": 0,
            "text": "A model is presented....",
            "end": 119,
            "tokens": [
                {
                    "begin": 0,
                    "text": "A",
                    "end": 1,
                    "id": "s0-t0"
                },
                {
                    "begin": 2,
                    "text": "model",
                    "end": 7,
                    "id": "s0-t1"
                },
                {
                    "begin": 8,
                    "text": "is",
                    "end": 10,
                    "id": "s0-t2"
                }....]}],
	"text": "A model is presented to characterize the class of languages obtained by adding reduplication to context-free languages....",
    "id": "c827f119-12ce-42db-8dce-f4a8ab1caa6c"
}
```

2. Train the model
3. Tune your model based on the results on validation set.
4. Call "predict" function to get results on raw text inputs tokenized into sentences.
```diff
import nltk
import nltk.data
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = "input text"

predict(neural_model, sent_tokenizer.tokenize(sentences))
```

