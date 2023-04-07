



<div align="center">

<img src="https://i.328888.xyz/2023/03/31/iwfiBd.png" width="400px">

**Large-scale, Informative, and Diverse Multi-round Dialogue Data**

<p align="center">
  <a href="http://39.101.77.220/">Data Explorer</a> •
  <a href="#data">Data</a> •
  <a href="#construction-of-ultrachat">Construction Process</a> •
  <a href="#training">Training</a> •
  <a href="#news">News and Future Plans</a>
</p>

</div>

<div align="center">

![Dialogues](https://img.shields.io/badge/Current\_Dialogues-570k-red?style=flat-square)
![Dialogues](https://img.shields.io/badge/Questions\_about\_the\_Wolrd-Released-green?style=flat-square)
![Dialogues](https://img.shields.io/badge/Writing\_and\_Creation-Unreleased-9cf?style=flat-square)
![Dialogues](https://img.shields.io/badge/Assistance\_on\_Existent\_Materials-Unreleased-9cf?style=flat-square)

</div>




This project aims to construct an *open-source, large-scale, and multi-round* dialogue data powered by Turbo APIs to facilitate the construction of powerful language models with general conversational capability.
In consideration of factors such as safeguarding privacy, **we do not directly use any data available on the Internet as prompts**.
To ensure generation quality, two separate ChatGPT Turbo APIs are adopted in generation, where one plays the role of the user to generate queries and the other generates the response. 
We instruct the user model with carefully designed prompt to mimic a human user behavior and call the two APIs iteratively. The generated dialogues undergo further post-processing and filtering.
<img align="bottom" src="https://i.328888.xyz/2023/03/31/iwIdSt.png" width="80px"> is composed of three sectors:

- 🌏 **Questions about the World**: The dialogue data in this sector is derived from a wide range of inquiries related to concepts, entities, and objects from the real world. The topics covered are extensive, spanning areas such as technology, art, and entrepreneurship.
- ✍🏻 **Writing and Creation**: The dialogue data in this sector is driven by the demands for writing/creation from scratch, and encompasses any tasks that an AI assistant may aid with in the creative process, spanning from email composition to crafting narratives and plays, and beyond.
- 📋 **Assistance on Existent Materials**: The dialogue data in this sector is generated based on existing materials, including but not limited to rewriting, continuation, summarization, and inference, covering a diverse range of topics.


Currently, we have released the Questions about the World sector, which contains 570k diverse and informative dialogues about the real world. Other data will be released in the future.

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
[Explore](http://39.101.77.220/) the data before downloading.

Download links:
- [Questions about the World [Part I + Part II]](https://cloud.tsinghua.edu.cn/f/a68feaa7cd124956a94d/?dl=1)

### Data Format
Each line in the downloaded data file is a json dict containing the data id and dialogue data in a list format. Below is an example line.

```
{
  "id": "0", 
  "data": [
    "How can cross training benefit groups like runners, swimmers, or weightlifters?", 
    "Cross training can benefit groups like runners, swimmers, or weightlifters in the following ways: ...", 
    "That makes sense. I've been wanting to improve my running time, but I never thought about incorporating strength training. Do you have any recommendations for specific exercises?", 
    "Sure, here are some strength training exercises that can benefit runners: ...", 
    "Hmm, I'm not really a fan of weightlifting though. Can I incorporate other forms of exercise into my routine to improve my running time?", 
    "Yes, absolutely! ...",
    ...
    ]
}

```


## Construction of UltraChat

The general idea of UltraChat is to use separate LLMs to generate opening lines, simulate users and respond to the queries.
Each sector of UltraChat has its own challenges and requires particular stragety designs. 
We will specify the construction process once a sector of UltraChat is released.


<details><summary> <b>Questions about the World</b> </summary>
<p>

#### Meta Topics & Sub-Topics

- The data is derived from 30 representative and diverse meta topics (icons are from [flaticon](https://www.flaticon.com/))

<div align="center">
<img src="https://i.328888.xyz/2023/04/01/i22Zoc.png" width="650px">
</div>

- Based on the above meta topics, we generate 1100+ subtopics for data construction
- For each subtopics, we generate up to 10 specific questions. 
- Then we use Turbo APIs to generate new relevant questions for each of the 10 questions. We use hand-crafted prompts to instruct the model to generate a diverse set of questions covering a wide range of common concepts and objects.
- For each question, we generate a 3~7-round dialogue using the two models iteratively as described above.

</p>

<p>

#### Common Real-world Entities

- We gather top-frequent 10000 named entities from Wikidata.
- We generate 5 meta questions for each entity using ChatGPT API.
- For each meta question, we generate 10 more specific questions and 20 related but general questions.
- We sample 20w specific questions and 25w general questions along with the 5w meta questions, and we generate a 3~7-round dialogue for each.

</p>

</details>

<details><summary> <b>Assistance on Existent Materials</b> </summary>
<p>
  
 
  
  - We will detail the construction method once this sector of data is released.
  
</p>
</details>

<details><summary> <b>Writing and Creation</b> </summary>
<p>
  
 - We will detail the construction method once this sector of data is released.
  
  
</p>
</details>


## Training
We provide a training script to fine-tune GPT-J on UltraChat in [`./train`](train).
- download the released data and put it under `./data`
- run `accelerate launch train.py` to start training

## News
- April 7, 2023: The second part of Questions about the World is released, it contains 290k generated multi-round dialogues.
- March 31, 2023: The first part of Questions about the World is released, it contains 280k generated multi-round dialogues.

## To Do
- [x] Release the rest part of data for Questions about the World.
- [ ] Continue to release the data of Writing and Creation and Assistance on Existent Materials in the future.
- [ ] Train a model on UltraChat and conduct in-detail analysis, welcome to use it to train your chat model!

## Citation
Feel free to cite the repo if you thinkd UltraChat is useful.

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
