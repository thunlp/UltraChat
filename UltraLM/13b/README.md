# Training LLaMA-13B

## Data Preparation
- download [UltraChat](https://github.com/thunlp/UltraChat#data-release) data in jsonline format and put under `../data/raw`
- run the script according to `../data/process.sh` to split the data into chunks with maximum length of 2048, save the processed data in `../data/processed`

## Train
- run `train.sh` to start training

## Template
For 13b model, we add `<pad>` as padding token and follow the template below

```
User: user input</s>
Assistant: LLM output</s>
```