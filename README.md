# ProtoTransformer Networks

A PyTorch implementation of *ProtoTransformer: A Meta-Learning Approach to Providing Student Feedback* (https://arxiv.org/abs/2107.14035). You can find a blog here (https://ai.stanford.edu/blog/prototransformer) and a press release here (https://www.nytimes.com/2021/07/20/technology/ai-education-neural-networks.html).

## Abstract

High-quality computer science education is limited by the difficulty of providing instructor feedback to students at scale. While this feedback could in principle be automated, supervised approaches to predicting the correct feedback are bottlenecked by the intractability of annotating large quantities of student code. In this paper, we instead frame the problem of providing feedback as few-shot classification, where a meta-learner adapts to give feedback to student code on a new programming question from just a few examples annotated by instructors. Because data for meta-training is limited, we propose a number of amendments to the typical few-shot learning framework, including task augmentation to create synthetic tasks, and additional side information to build stronger priors about each task. These additions are combined with a transformer architecture to embed discrete sequences (e.g. code) to a prototypical representation of a feedback class label. On a suite of few-shot natural language processing tasks, we match or outperform state-of-the-art performance. Then, on a collection of student solutions to exam questions from an introductory university course, we show that our approach reaches an average precision of 88% on unseen questions, surpassing the 82% precision of teaching assistants. Our approach was successfully deployed to deliver feedback to 16,000 student exam-solutions in a programming course offered by a tier 1 university. This is, to the best of our knowledge, the first successful deployment of a machine learning based feedback to open-ended student code.

### Main Intuition

With the rise of large online computer science courses, there is an abundance of high-quality content. At the same time, the sheer size of these courses makes high-quality feedback to student work more and more difficult. Several computational approaches have been proposed to automatically produce personalized feedback, but each falls short. We proposed a new AI system based on meta-learning that trains a neural network to ingest student code and output feedback. On a dataset of student solutions to Stanford’s CS106A exams, we found the AI system to match human instructors in feedback quality.  

## Setup / Installation

We use Python 3, PyTorch 1.7.1, PyTorch Lightning 1.1.8, and a conda environment. Consider a variation of the commands below:

```
conda create -n proto python=3 anaconda
conda activate proto
conda install pytorch=1.7.1 torchvision -c pytorch
pip install tqdm dotmap
```

## Data

There are a number of datasets used in the NLP experiments.

- From [Geng et. al.](https://arxiv.org/abs/1902.10482v2), we have a few-shot NLP dataset built from the Amazon corpus. We downloaded the data from [this](https://github.com/zhongyuchen/few-shot-text-classification) public repo.
- From [Bao et. al.](https://arxiv.org/pdf/1908.06039), we use the suite of few-shot text datasets spanning news, rcv1, reuters, etc. The data can be found [here](https://github.com/YujiaBao/Distributional-Signatures).
- We repurpose the 20-newsgroups dataset for a few-shot topic classification task. You can download the raw 20-newsgroups datasets [here](http://qwone.com/~jason/20Newsgroups/).

Unfortunately, we cannot release the dataset of student responses to university level programming assignments due to privacy concerns. We are working towards a publically shareably version in the near future.

## Usage

For every fresh terminal instance, you should run

```
source init_env.sh
```

to add the correct paths to `sys.path` before running anything else.

## Citation

If you find this useful for your research, please cite:

```
@article{wu2021prototransformer,
  title={ProtoTransformer: A Meta-Learning Approach to Providing Student Feedback},
  author={Wu, Mike and Goodman, Noah and Piech, Chris and Finn, Chelsea},
  journal={arXiv preprint arXiv:2107.14035},
  year={2021}
}
```