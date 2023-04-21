



<div align="center">

<img src="https://i.328888.xyz/2023/03/31/iwfiBd.png" width="400px">

**Large-scale, Informative, and Diverse Multi-round Dialogue Data**

<p align="center">
  <a href="http://39.101.77.220/">Data Explorer</a> •
  <a href="https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac">Nomic AI Atlas Explorer</a> •
  <a href="#data">Data Release</a> •
  <a href="#construction-of-ultrachat">Construction Process</a> •
  <a href="#training">Training</a> •
  <a href="#news">News and Future Plans</a>
</p>

</div>

<div align="center">

![Dialogues](https://img.shields.io/badge/Current\_Dialogues-1.57M-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Questions\_about\_the\_Wolrd-Released-green?style=flat-square)
![Dialogues](https://img.shields.io/badge/Writing\_and\_Creation-Released-green?style=flat-square)
![Dialogues](https://img.shields.io/badge/Assistance\_on\_Existent\_Materials-Released-green?style=flat-square)

</div>




This project aims to construct *open-source, large-scale, and multi-round* dialogue data powered by Turbo APIs to facilitate the construction of powerful language models with general conversational capability.
In consideration of factors such as safeguarding privacy, **we do not directly use any data available on the Internet as prompts**.
To ensure generation quality, two separate ChatGPT Turbo APIs are adopted in generation, where one plays the role of the user to generate queries and the other generates the response. 
We instruct the user model with carefully designed prompts to mimic human user behavior and call the two APIs iteratively. The generated dialogues undergo further post-processing and filtering.
<img align="bottom" src="https://i.328888.xyz/2023/03/31/iwIdSt.png" width="80px"> is composed of three sectors:

- 🌏 **Questions about the World**: The dialogue data in this sector is derived from a wide range of inquiries related to concepts, entities, and objects from the real world. The topics covered are extensive, spanning areas such as technology, art, and entrepreneurship.
- ✍🏻 **Writing and Creation**: The dialogue data in this sector is driven by the demands for writing/creation from scratch, and encompasses any tasks that an AI assistant may aid within the creative process, spanning from email composition to crafting narratives and plays, and beyond.
- 📋 **Assistance on Existent Materials**: The dialogue data in this sector is generated based on existing materials, including but not limited to rewriting, continuation, summarization, and inference, covering a diverse range of topics.



<details><summary> <b>An Example of UltraChat </b> </summary>
<p>
 <div align="center">
 <img src="https://i.328888.xyz/2023/04/02/iHh8DC.png" width="900px">
 </div>
</p>
</details>


## Data

The dataset is intended solely for research and educational purposes and should not be construed as reflecting the opinions or views of the creators, owners, or contributors of this dataset. And it is distributed under [CC BY NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use).


### Data Release
[Explore](http://39.101.77.220/) the data before downloading, or use [Atlas explorer](https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac).

- 🤗 [Huggingface Datasets Host](https://huggingface.co/datasets/stingning/ultrachat)

Direct Download links:
- [Questions about the World [Part I + Part II]](https://cloud.tsinghua.edu.cn/f/0a27393192ad46a5a081/?dl=1)
- [Writing and Creation [Part I]](https://cloud.tsinghua.edu.cn/f/57258a87846243218a9b/?dl=1)
- [Writing and Creation [Part II]](https://cloud.tsinghua.edu.cn/f/099b4dd71b82448fb7fb/?dl=1)
- [Assistance on Existent Materials [Part I]](https://cloud.tsinghua.edu.cn/f/1f7abdf2d2564cb4b338/?dl=1)

### Data Format
Each line in the downloaded data file is a json dict containing the data id and dialogue data in a list format. Below is an example line.

```JSON
{
  "id": "0", 
  "data": [
    "How can cross training benefit groups like runners, swimmers, or weightlifters?", 
    "Cross training can benefit groups like runners, swimmers, or weightlifters in the following ways: ...", 
    "That makes sense. I've been wanting to improve my running time, but I never thought about incorporating strength training. Do you have any recommendations for specific exercises?", 
    "Sure, here are some strength training exercises that can benefit runners: ...", 
    "Hmm, I'm not really a fan of weightlifting though. Can I incorporate other forms of exercise into my routine to improve my running time?", 
    "Yes, absolutely! ...",
    "..."
    ]
}

```


## Construction of UltraChat

The general idea of UltraChat is to use separate LLMs to generate opening lines, simulate users and respond to queries.
Each sector of UltraChat has its own challenges and requires particular strategy designs. 
We will specify the construction process once a sector of UltraChat is released.


<details><summary> <b>Questions about the World</b> </summary>
<p>

#### Meta Topics & Sub-Topics

- The data is derived from 30 representative and diverse meta topics (icons are from [flaticon](https://www.flaticon.com/))

<div align="center">
<img src="https://i.328888.xyz/2023/04/01/i22Zoc.png" width="650px">
</div>

- Based on the above meta topics, we generate 1100+ subtopics for data construction
- For each subtopic, we generate up to 10 specific questions. 
- Then we use Turbo APIs to generate new relevant questions for each of the 10 questions. We use hand-crafted prompts to instruct the model to generate a diverse set of questions covering a wide range of common concepts and objects.
- For each question, we generate a 3~7-round dialogue using the two models iteratively as described above.

</p>

<p>

#### Common Real-world Entities

- We gather top-frequent 10000 named entities from Wikidata.
- We generate 5 meta questions for each entity using ChatGPT API.
- For each meta question, we generate 10 more specific questions and 20 related but general questions.
- We sample 200k specific questions and 250k general questions along with the 50k meta-questions, and we generate a 3~7-round dialogue for each.

</p>

</details>

<details><summary> <b>Writing and Creation</b> </summary>
<p>

- We first collect 20 types of writing, as shown above.
- For each type of writing, generate 200 different instructions that ask an AI assistant to generate text material, and 80% of the instructions are further expanded and detailed.
- Use the generated instructions as initial input and generate a 2~4-round dialogue each.
  
</p>
</details>

<details><summary> <b>Assistance on Existent Materials</b> </summary>
<p>

- We extract ~10w diverse materials from C4 dataset.
- We generate up to 5 questions/instructions for each piece of material.
- We combine the material with each question/instruction with a set of manually designed template as the initial input of a user to start a dialogue with AI assistant.
- For each input, we generate a 2~4-round dialogue.
 
</p>
</details>




## Training
We provide a training script to fine-tune GPT-J on UltraChat in [`./train`](train), which is implemented with [OpenPrompt](https://github.com/thunlp/OpenPrompt)
- Download the released data and put it under `./data`
- Run `accelerate launch train.py` to start training

## News
- April 20, 2023: Released all data, more processing and additional data are expected.
- April 17, 2023: The rest of the Writing and Creation sector is released (457k). 
- April 12, 2023: The first part of the Writing and Creation sector is released.
- April 9, 2023: Supported by [gpt4all](https://github.com/nomic-ai/gpt4all), we now also have an [Atlas explorer](https://atlas.nomic.ai/map/0ce65783-c3a9-40b5-895d-384933f50081/a7b46301-022f-45d8-bbf4-98107eabdbac)
- April 8, 2023: We release a training script by taking GPT-J as an example.
- April 7, 2023: The second part of Questions about the World is released. It contains 290k generated multi-round dialogues.
- March 31, 2023: The first part of Questions about the World is released. It contains 280k generated multi-round dialogues.

## To Do
- [x] Release the rest part of the data for Questions about the World.
- [x] Continue to release the data of Writing and Creation.
- [x] Continue to release the data of  Assistance on Existent Materials in the future.
- [ ] Train a model on UltraChat and conduct in-detail analysis. Welcome to use it to train your chat model!
- [ ] There will be a Chinese version of UltraChat.


## Limitations

- Auto-generated data may contain hallucination and other formats of false facts. This issue will mainly appear in the first sector of the data. 
- To address the issue, more extensive post-processing will be conducted.

## Citation
Feel free to cite the repo if you think UltraChat is useful.

```bibtex
@misc{UltraChat,
  author = {Ding, Ning and Chen, Yulin and Xu, Bokai and Hu, Shengding and Qin, Yujia and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen},
  title = {UltraChat: A Large-scale Auto-generated Multi-round Dialogue Data},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thunlp/ultrachat}},
}
```
