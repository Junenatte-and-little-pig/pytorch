# -*- encoding: utf-8 -*-
import torch


def clean_words(line):
    punctuation=',./!?<>-=_+[]{}\\|;:"“”‘’'
    word_list=line.lower().replace('\n',' ').split()
    word_list=[word.strip(punctuation) for word in word_list]
    return word_list


def main():
    with open('../data/p1ch4/jane-austen/1342-0.txt',encoding='utf8') as f:
        text=f.read()
    # one-hot for every letter
    # lines=text.split('\n')
    # line=lines[200]
    # print(line)
    # letter_t=torch.zeros(len(line),128)
    # for i , letter in enumerate(line.lower().strip()):
    #     index=ord(letter) if ord(letter) <128 else 0
    #     letter_t[i][index]=1

    word_list=sorted(set(clean_words(text)))
    word_dict={word:i for i, word in enumerate(word_list)}
    print(word_list)
    print(word_dict)

    word_t=torch.zeros(len(word_list),len(word_dict))
    for i , word in enumerate(word_list):
        index=word_dict[word]
        word_t[i][index]=1
    print(word_t)



if __name__ == '__main__':
    main()
