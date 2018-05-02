///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rnnlmlib.h"
#include "hierarchical_cluster_fsthistory.h"


///// fast exp() implementation
static union
{
    double d;
    struct
    {
        int j,i;
    } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)

/*#define BOUND_A 0.3
#define VALUE_A 0.1
#define BOUND_B 0.3
#define VALUE_B 0.2
#define BOUND_C 0.7
#define VALUE_C 0.5
#define VALUE_OTHER 0.6*/

// #define BOUND_A 0.07
// #define VALUE_A 0.00
// #define BOUND_B 0.3
// #define VALUE_B 0.2
// #define BOUND_C 0.7
// #define VALUE_C 0.5
// #define VALUE_OTHER 0.5

/* blas的全称是basic linear algebra subprograms,用于向量和矩阵计算的高性能数学库， 
blas本身是Fortran写的,cblas是blas的c语言接口库，rnnlmlib.cpp文件本身是用c++写的,
需要调用c语言的cblas,所以需要用extern "C"来表明{}里面的内容需要按c语言的规范进行编
译和链接，这是因为C＋＋和C程序编译完成后在目标代码中命名规则不同,extern "C"实现了c
和c++的混合编程。*/
///// include blas
#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif
//

void CRnnLM::setDiscretizer(Discretizer *dis) {
	disc_map_set = 1;
	d = dis;
	fsth = new HierarchicalClusterFstHistory();
}

//个生成随机小数的函数
real CRnnLM::random(real min, real max)
{
    return rand()/(real)RAND_MAX*(max-min)+min;
}

void CRnnLM::setTrainFile(char *str)
{
    strcpy(train_file, str);
}

void CRnnLM::setValidFile(char *str)
{
    strcpy(valid_file, str);
}

void CRnnLM::setTestFile(char *str)
{
    strcpy(test_file, str);
}

void CRnnLM::setRnnLMFile(char *str)
{
    strcpy(rnnlm_file, str);
}


/* 
功能：从文件中读取一个单词到word
注意：
1.单词最长不能超过99(最后一个字符得为'\0')，否则会被截断
2.训练集中每个句子结尾都会自动生成</s>作为一个单独的词，被复制到word返回，这在后面也是用来判断一个句子是否结束的标志。
*/
void CRnnLM::readWord(char *word, FILE *fin)
{
    int a=0, ch;

    while (!feof(fin)) 
    {
        ch=fgetc(fin);
        
        if (ch==13) 
            continue;

        if ((ch==' ') || (ch=='\t') || (ch=='\n')) 
        {
    	    if (a>0) 
            {
                if (ch=='\n') ungetc(ch, fin);
                break;
            }

            if (ch=='\n') 
            {
                strcpy(word, (char *)"</s>");
                return;
            }
            else 
                continue;
        }

        word[a]=ch;
        a++;

        if (a>=MAX_STRING) 
        {
            //printf("Too long word found!\n");   //truncate too long words
            a--;
        }
    }
    word[a]=0;
}

int CRnnLM::getWordHash(char *word)
{
    unsigned int hash, a;
    
    hash=0;
    for (a=0; a<strlen(word); a++) 
        hash=hash*237+word[a];
    hash=hash%vocab_hash_size;
    
    return hash;
}

int CRnnLM::searchVocab(char *word)
{
    int a;
    unsigned int hash;
    
    hash=getWordHash(word);
    
    if (vocab_hash[hash]==-1) 
        return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) 
        return vocab_hash[hash];
    
    for (a=0; a<vocab_size; a++) 
    {	//search in vocabulary
        if (!strcmp(word, vocab[a].word)) 
        {
    	    vocab_hash[hash]=a;
    	    return a;
    	}
    }

    return -1;							//return OOV if not found
}

/*
读取当前文件指针所指的单词,并返回该单词在vocab中的索引，注意无论是训练数据、
验证数据、测试数据文件的格式都是文件末尾空行，所以按照文件内容顺序查找，查找
到文件末尾一定是</s>，然后fin就到文件末尾了。
*/
int CRnnLM::readWordIndex(FILE *fin)
{
    char word[MAX_STRING];

    readWord(word, fin);
    if (feof(fin)) return -1;

    return searchVocab(word);
}

int CRnnLM::addWordToVocab(char *word)
{
    unsigned int hash;
    
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn=0;
    vocab_size++;

    if (vocab_size+2>=vocab_max_size)
    {   //realloc是用来扩大或缩小内存的,扩大时原来的内容不变,系统直接  
        //在后面找空闲内存,如果没找到，则会把前面的数据重新移动到一个够大的地方  
        //即realloc可能会导致数据的移动,这算自己顺便看源码边复习一些c的知识吧 
        vocab_max_size+=100;
        vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    
    hash=getWordHash(word);
    vocab_hash[hash]=vocab_size-1;

    return vocab_size-1;
}

/*
择排序的算法，将vocab[1]到vocab[vocab_size-1]按照他们出现的频数从大到小排序
*/
void CRnnLM::sortVocab()
{
    int a, b, max;
    vocab_word swap;
    
    for (a=1; a<vocab_size; a++) 
    {
        max=a;
        for (b=a+1; b<vocab_size; b++) 
            if (vocab[max].cn<vocab[b].cn) 
                max=b;

        swap=vocab[max];
        vocab[max]=vocab[a];
        vocab[a]=swap;
    }
}

void CRnnLM::learnVocabFromTrainFile()    //assumes that vocabulary is empty
{
    char word[MAX_STRING];
    FILE *fin;
    int a, i, train_wcn;
    
    for (a=0; a<vocab_hash_size; a++) 
        vocab_hash[a]=-1;

    fin=fopen(train_file, "rb");

    vocab_size=0;

    addWordToVocab((char *)"</s>");

    train_wcn=0;
    while (1) 
    {
        readWord(word, fin);
        if (feof(fin)) 
            break;
        
        train_wcn++;

        i=searchVocab(word);
        if (i==-1) //word out of vocab
        {
            a=addWordToVocab(word);
            vocab[a].cn=1;
        } 
        else 
            vocab[i].cn++;
    }

    sortVocab();
    
    //select vocabulary size
    /*a=0;
    while (a<vocab_size) {
	a++;
	if (vocab[a].cn==0) break;
    }
    vocab_size=a;*/

    if (debug_mode>0) 
    {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %d\n", train_wcn);
    }
    
    train_words=train_wcn;

    fclose(fin);
}

/*
为什么会建立压缩层,见论文EXTENSIONS OF RECURRENT NEURAL NETWORK LANGUAGE MODEL，
里面说的压缩层是为了减少输出到隐层的参数,并且减小了总的计算复杂度，至于为什么增加压
缩层能够使计算量减小,暂时还不明白。？？？？？
*/
void CRnnLM::saveWeights()      //saves current weights and unit activations
{
    int a,b;

    //暂存输入层神经元值  
    for (a=0; a<layer0_size; a++) 
    {
        neu0b[a].ac=neu0[a].ac;
        neu0b[a].er=neu0[a].er;
    }

    for (a=0; a<layer1_size; a++) 
    {
        neu1b[a].ac=neu1[a].ac;
        neu1b[a].er=neu1[a].er;
    }
    
    for (a=0; a<layerc_size; a++) 
    {
        neucb[a].ac=neuc[a].ac;
        neucb[a].er=neuc[a].er;
    }
    
    for (a=0; a<layer2_size; a++) 
    {
        neu2b[a].ac=neu2[a].ac;
        neu2b[a].er=neu2[a].er;
    }
    
    for (b=0; b<layer1_size; b++)
    {
        for (a=0; a<layer0_size; a++) 
        {
            syn0b[a+b*layer0_size].weight=syn0[a+b*layer0_size].weight;
        }
    }
        
    
    if (layerc_size>0) 
    {
        for (b=0; b<layerc_size; b++) 
            for (a=0; a<layer1_size; a++) 
            {
                syn1b[a+b*layer1_size].weight=syn1[a+b*layer1_size].weight;
            }
        
        for (b=0; b<layer2_size; b++) 
            for (a=0; a<layerc_size; a++) 
            {
                syncb[a+b*layerc_size].weight=sync[a+b*layerc_size].weight;
            }
    }
    else 
    {
	    for (b=0; b<layer2_size; b++) 
            for (a=0; a<layer1_size; a++) 
            {
                syn1b[a+b*layer1_size].weight=syn1[a+b*layer1_size].weight;
            }
    }
    
    //for (a=0; a<direct_size; a++) syn_db[a].weight=syn_d[a].weight;
}

void CRnnLM::restoreWeights()      //restores current weights and unit activations from backup copy
{
    int a,b;

    for (a=0; a<layer0_size; a++) {
        neu0[a].ac=neu0b[a].ac;
        neu0[a].er=neu0b[a].er;
    }

    for (a=0; a<layer1_size; a++) {
        neu1[a].ac=neu1b[a].ac;
        neu1[a].er=neu1b[a].er;
    }
    
    for (a=0; a<layerc_size; a++) {
        neuc[a].ac=neucb[a].ac;
        neuc[a].er=neucb[a].er;
    }
    
    for (a=0; a<layer2_size; a++) {
        neu2[a].ac=neu2b[a].ac;
        neu2[a].er=neu2b[a].er;
    }

    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
        syn0[a+b*layer0_size].weight=syn0b[a+b*layer0_size].weight;
    }
    
    if (layerc_size>0) {
	for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=syn1b[a+b*layer1_size].weight;
	}
	
	for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
	    sync[a+b*layerc_size].weight=syncb[a+b*layerc_size].weight;
	}
    }
    else {
	for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=syn1b[a+b*layer1_size].weight;
	}
    }
    
    //for (a=0; a<direct_size; a++) syn_d[a].weight=syn_db[a].weight;
}

void CRnnLM::saveContext()		//useful for n-best list processing
{
    int a;
    
    for (a=0; a<layer1_size; a++) 
        neu1b[a].ac=neu1[a].ac;
}

void CRnnLM::restoreContext()
{
    int a;
    
    for (a=0; a<layer1_size; a++) 
        neu1[a].ac=neu1b[a].ac;
}

void CRnnLM::saveContext2()
{
    int a;
    
    for (a=0; a<layer1_size; a++) 
        neu1b2[a].ac=neu1[a].ac;
}

void CRnnLM::restoreContext2()
{
    int a;
    
    for (a=0; a<layer1_size; a++) 
        neu1[a].ac=neu1b2[a].ac;
}

/*
功能：初始化网络
主要是完成分配内存，初始化等工作，这个过程也就相当于把网络给搭建起来
上面的函数初始化网络涉及最大熵模型，即可以简单的理解为输入层到输出层的直接连接，
虽然作者在论文中总是强调可以这么认为，但我觉的并不是那样简单的直接连接着，中间
会有一个历史数组。
*/
void CRnnLM::initNet()
{
    int a, b, cl;

    //layer1_size初始为30 
    //class_size初始时为100
    layer0_size=vocab_size+layer1_size;
    layer2_size=vocab_size+class_size;

    neu0=(struct neuron *)calloc(layer0_size, sizeof(struct neuron));
    neu1=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
    neuc=(struct neuron *)calloc(layerc_size, sizeof(struct neuron));
    neu2=(struct neuron *)calloc(layer2_size, sizeof(struct neuron));

    syn0=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
    if (layerc_size==0)
	    syn1=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    else 
    {
        syn1=(struct synapse *)calloc(layer1_size*layerc_size, sizeof(struct synapse));
        sync=(struct synapse *)calloc(layerc_size*layer2_size, sizeof(struct synapse));
    }

    if (syn1==NULL) 
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    if (layerc_size>0) 
        if (sync==NULL) 
        {
            printf("Memory allocation failed\n");
            exit(1);
        }
    
    //建立输入层到输出层的参数,direct_size是long long类型的,由-direct参数指定,单位是百万  
    //比如-direct传进来的是2，则真实的direct_size = 2*10^6  
    //如果输入层和输出层之间有直连边，该变换同样需要一个参数矩阵W（W的规模为|V|*m(n-1)）:y=b + Wx + Utanh(d + Hx).
    syn_d=(direct_t *)calloc((long long)direct_size, sizeof(direct_t));

    if (syn_d==NULL) 
    {
        printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)direct_size * (long long)sizeof(direct_t));
        exit(1);
    }

    //创建神经元备份空间  
    neu0b=(struct neuron *)calloc(layer0_size, sizeof(struct neuron));
    neu1b=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
    neucb=(struct neuron *)calloc(layerc_size, sizeof(struct neuron));
    neu1b2=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
    neu2b=(struct neuron *)calloc(layer2_size, sizeof(struct neuron));

    //创建突触(即权值参数)的备份空间  
    syn0b=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
    //syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    if (layerc_size==0)
	    syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    else 
    {
        syn1b=(struct synapse *)calloc(layer1_size*layerc_size, sizeof(struct synapse));
        syncb=(struct synapse *)calloc(layerc_size*layer2_size, sizeof(struct synapse));
    }

    if (syn1b==NULL) 
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    for (a=0; a<layer0_size; a++) 
    {
        neu0[a].ac=0;
        neu0[a].er=0;
    }

    for (a=0; a<layer1_size; a++) 
    {
        neu1[a].ac=0;
        neu1[a].er=0;
    }
    
    for (a=0; a<layerc_size; a++) 
    {
        neuc[a].ac=0;
        neuc[a].er=0;
    }
    
    for (a=0; a<layer2_size; a++) 
    {
        neu2[a].ac=0;
        neu2[a].er=0;
    }

    for (b=0; b<layer1_size; b++) 
        for (a=0; a<layer0_size; a++) 
        {
            syn0[a+b*layer0_size].weight=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
        }

    if (layerc_size>0) {
	for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
	
	for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
	    sync[a+b*layerc_size].weight=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
    }
    else {
	for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
    }
    
    //输入到输出直连的参数初始化为0 
    long long aa;
    for (aa=0; aa<direct_size; aa++) 
        syn_d[aa]=0;
    
    if (bptt>0) 
    {
        //初始化bptt_history,bptt+bptt_block初始化为-1，后10个int由于是calloc申请，默认为0
        bptt_history=(int *)calloc((bptt+bptt_block+10), sizeof(int));
        for (a=0; a<bptt+bptt_block; a++) 
            bptt_history[a]=-1;
        //
        bptt_hidden=(neuron *)calloc((bptt+bptt_block+1)*layer1_size, sizeof(neuron));
        for (a=0; a<(bptt+bptt_block)*layer1_size; a++) 
        {
            bptt_hidden[a].ac=0;
            bptt_hidden[a].er=0;
        }
        //
        bptt_syn0=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
        if (bptt_syn0==NULL) 
        {
            printf("Memory allocation failed\n");
            exit(1);
        }
    }

    //saveWeights里面并没有保存输入层到输出层的参数，即syn_d
    saveWeights();
    
    double df, dd;
    int i;
    
    df=0;
    dd=0;
    a=0;
    b=0;

    /*
    对单词分类，注意下面的vocab是从大到小排好序的，下面都是对word进行分类,
    分类的依据就是他们的一元词频，分类的最终结果就是越靠近前面类别的word很少,
    他们出现的频数比较高，越靠近后面的类别所包含的word就非常多,他们在语料中出
    现比较稀疏。
    */
    if (old_classes) 
    {  	// old classes
        for (i=0; i<vocab_size; i++) 
            b+=vocab[i].cn;
        for (i=0; i<vocab_size; i++) 
        {
            df+=vocab[i].cn/(double)b;
            if (df>1) df=1;
            if (df>(a+1)/(double)class_size) 
            {
    	        vocab[i].class_index=a;
    	        if (a<class_size-1) 
                    a++;
            } 
            else 
            {
    	        vocab[i].class_index=a;
            }
        }
    } 
    else 
    {			// new classes
        //统计所有词汇频次和
        for (i=0; i<vocab_size; i++) 
            b+=vocab[i].cn;
        for (i=0; i<vocab_size; i++) 
            dd+=sqrt(vocab[i].cn/(double)b);
        for (i=0; i<vocab_size; i++)    
        {
	        df+=sqrt(vocab[i].cn/(double)b)/dd;
            if (df>1) 
                df=1;
            if (df>(a+1)/(double)class_size) 
            {
    	        vocab[i].class_index=a;
    	        if (a<class_size-1) 
                    a++;
            } 
            else 
            {
    	        vocab[i].class_index=a;
            }
	    }
    }
    
    //allocate auxiliary class variables (for faster search when normalizing probability at output layer)
    //下面是为了加速查找,最终达到的目的就是给定一个类别，能很快的遍历得到该类别的所有word
    class_words=(int **)calloc(class_size, sizeof(int *));
    class_cn=(int *)calloc(class_size, sizeof(int));
    class_max_cn=(int *)calloc(class_size, sizeof(int));
    
    for (i=0; i<class_size; i++) 
    {
        class_cn[i]=0;
        class_max_cn[i]=10;
        class_words[i]=(int *)calloc(class_max_cn[i], sizeof(int));
    }
    
    for (i=0; i<vocab_size; i++) 
    {
        cl=vocab[i].class_index;
        class_words[cl][class_cn[cl]]=i;
        class_cn[cl]++;
        if (class_cn[cl]+2>=class_max_cn[cl]) 
        {
            class_max_cn[cl]+=10;
            class_words[cl]=(int *)realloc(class_words[cl], class_max_cn[cl]*sizeof(int));
        }
    }
}

void CRnnLM::saveNet()       //will save the whole network structure                                                        
{
    FILE *fo;
    int a, b;
    char str[1000];
    float fl;
    
    sprintf(str, "%s.temp", rnnlm_file);

    fo=fopen(str, "wb");
    if (fo==NULL) 
    {
        printf("Cannot create file %s\n", rnnlm_file);
        exit(1);
    }
    fprintf(fo, "version: %d\n", version);
    fprintf(fo, "file format: %d\n\n", filetype);

    fprintf(fo, "training data file: %s\n", train_file);
    fprintf(fo, "validation data file: %s\n\n", valid_file);

    fprintf(fo, "last probability of validation data: %f\n", llogp);
    fprintf(fo, "number of finished iterations: %d\n", iter);

    fprintf(fo, "current position in training data: %d\n", train_cur_pos);
    fprintf(fo, "current probability of training data: %f\n", logp);
    fprintf(fo, "save after processing # words: %d\n", anti_k);
    fprintf(fo, "# of training words: %d\n", train_words);

    fprintf(fo, "input layer size: %d\n", layer0_size);
    fprintf(fo, "hidden layer size: %d\n", layer1_size);
    fprintf(fo, "compression layer size: %d\n", layerc_size);
    fprintf(fo, "output layer size: %d\n", layer2_size);

    fprintf(fo, "direct connections: %lld\n", direct_size);
    fprintf(fo, "direct order: %d\n", direct_order);
    
    fprintf(fo, "bptt: %d\n", bptt);
    fprintf(fo, "bptt block: %d\n", bptt_block);
    
    fprintf(fo, "vocabulary size: %d\n", vocab_size);
    fprintf(fo, "class size: %d\n", class_size);
    
    fprintf(fo, "old classes: %d\n", old_classes);
    fprintf(fo, "independent sentences mode: %d\n", independent);
    
    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
    fprintf(fo, "current learning rate: %f\n", alpha);
    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
    fprintf(fo, "\n");

    fprintf(fo, "\nVocabulary:\n");
    for (a=0; a<vocab_size; a++) 
        fprintf(fo, "%6d\t%10d\t%s\t%d\n", a, vocab[a].cn, vocab[a].word, vocab[a].class_index);

    
    if (filetype==TEXT) 
    {
        fprintf(fo, "\nHidden layer activation:\n");
        for (a=0; a<layer1_size; a++) 
            fprintf(fo, "%.4f\n", neu1[a].ac);
    }
    if (filetype==BINARY) 
    {
    	for (a=0; a<layer1_size; a++) 
        {
    	    fl=neu1[a].ac;
    	    fwrite(&fl, 4, 1, fo);
    	}
    }
    //////////
    if (filetype==TEXT) 
    {
        fprintf(fo, "\nWeights 0->1:\n");
        for (b=0; b<layer1_size; b++) 
        {
            for (a=0; a<layer0_size; a++) 
            {
                fprintf(fo, "%.4f\n", syn0[a+b*layer0_size].weight);
            }
        }
    }
    if (filetype==BINARY) 
    {
        for (b=0; b<layer1_size; b++) 
        {
            for (a=0; a<layer0_size; a++) 
            {
                fl=syn0[a+b*layer0_size].weight;
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    /////////
    if (filetype==TEXT) {
	if (layerc_size>0) {
	    fprintf(fo, "\n\nWeights 1->c:\n");
	    for (b=0; b<layerc_size; b++) {
		for (a=0; a<layer1_size; a++) {
    		    fprintf(fo, "%.4f\n", syn1[a+b*layer1_size].weight);
    		}
    	    }
    	
    	    fprintf(fo, "\n\nWeights c->2:\n");
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layerc_size; a++) {
    		    fprintf(fo, "%.4f\n", sync[a+b*layerc_size].weight);
    		}
    	    }
	}
	else
	{
	    fprintf(fo, "\n\nWeights 1->2:\n");
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layer1_size; a++) {
    		    fprintf(fo, "%.4f\n", syn1[a+b*layer1_size].weight);
    		}
    	    }
    	}
    }
    if (filetype==BINARY) {
	if (layerc_size>0) {
	    for (b=0; b<layerc_size; b++) {
		for (a=0; a<layer1_size; a++) {
		    fl=syn1[a+b*layer1_size].weight;
    		    fwrite(&fl, 4, 1, fo);
    		}
    	    }
    	
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layerc_size; a++) {
    		    fl=sync[a+b*layerc_size].weight;
    		    fwrite(&fl, 4, 1, fo);
    		}
    	    }
	}
	else
	{
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layer1_size; a++) {
    		    fl=syn1[a+b*layer1_size].weight;
    		    fwrite(&fl, 4, 1, fo);
    		}
    	    }
    	}
    }
    ////////
    if (filetype==TEXT) {
	fprintf(fo, "\nDirect connections:\n");
	long long aa;
	for (aa=0; aa<direct_size; aa++) {
    	    fprintf(fo, "%.2f\n", syn_d[aa]);
	}
    }
    if (filetype==BINARY) {
	long long aa;
	for (aa=0; aa<direct_size; aa++) {
    	    fl=syn_d[aa];
    	    fwrite(&fl, 4, 1, fo);
    	    
    	    /*fl=syn_d[aa]*4*256;			//saving direct connections this way will save 50% disk space; several times more compression is doable by clustering
    	    if (fl>(1<<15)-1) fl=(1<<15)-1;
    	    if (fl<-(1<<15)) fl=-(1<<15);
    	    si=(signed short int)fl;
    	    fwrite(&si, 2, 1, fo);*/
	}
    }
    ////////    
    fclose(fo);
    
    //最后将名字更改为指定的rnnlm_file，那为啥最开始要改呢?  
    //这里不太明白 
    rename(str, rnnlm_file);
}

//从文件流中读取一个字符使其ascii等于delim  
//随后文件指针指向delim的下一个
void CRnnLM::goToDelimiter(int delim, FILE *fi)
{
    int ch=0;

    while (ch!=delim) 
    {
        ch=fgetc(fi);
        if (feof(fi)) 
        {
            printf("Unexpected end of file\n");
            exit(1);
        }
    }
}

void CRnnLM::restoreNet()    //will read whole network structure
{
    FILE *fi;
    int a, b, ver;
    float fl;
    char str[MAX_STRING];
    double d;

    fi=fopen(rnnlm_file, "rb");
    if (fi==NULL) 
    {
        printf("ERROR: model file '%s' not found!\n", rnnlm_file);
        exit(1);
    }

    goToDelimiter(':', fi);
    fscanf(fi, "%d", &ver);
    if ((ver==4) && (version==5)) 
    /* we will solve this later.. */ 
        ; 
    else if (ver!=version) 
    {
        printf("Unknown version of file %s\n", rnnlm_file);
        exit(1);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &filetype);
    //
    goToDelimiter(':', fi);
    if (train_file_set==0) 
    {
	    fscanf(fi, "%s", train_file);
    } 
    else 
        fscanf(fi, "%s", str);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", valid_file);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &llogp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &iter);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_cur_pos);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &logp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &anti_k);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_words);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer0_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer1_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layerc_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer2_size);
    //
    if (ver>5) 
    {
        goToDelimiter(':', fi);
        fscanf(fi, "%lld", &direct_size);
    }
    //
    if (ver>6) 
    {
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &direct_order);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt);
    //
    if (ver>4) 
    {
        goToDelimiter(':', fi);
        fscanf(fi, "%d", &bptt_block);
    } 
    else 
        bptt_block=10;
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &vocab_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &class_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &old_classes);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &independent);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &d);
    starting_alpha=d;
    //
    goToDelimiter(':', fi);
    if (alpha_set==0) 
    {
        fscanf(fi, "%lf", &d);
        alpha=d;
    } 
    else 
        fscanf(fi, "%lf", &d);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &alpha_divide);
    //
    
    
    //read normal vocabulary
    //下面是把vocab从train_file中恢复过来
    if (vocab_max_size<vocab_size) 
    {
        if (vocab!=NULL) 
            free(vocab);
        vocab_max_size=vocab_size+1000;
        vocab=(struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));    //initialize memory for vocabulary
    }
    //
    goToDelimiter(':', fi);
    for (a=0; a<vocab_size; a++) 
    {
        //fscanf(fi, "%d%d%s%d", &b, &vocab[a].cn, vocab[a].word, &vocab[a].class_index);
        fscanf(fi, "%d%d", &b, &vocab[a].cn);
        readWord(vocab[a].word, fi);
        fscanf(fi, "%d", &vocab[a].class_index);
        //printf("%d  %d  %s  %d\n", b, vocab[a].cn, vocab[a].word, vocab[a].class_index);
    }
    //
    if (neu0==NULL) 
        initNet();		//memory allocation here
    //
    
    //hiddern layer activation data
    //由于对网络的权值分为两种模式,所以这里也应该分情况读入  
    //对于大量的实数，二进制模式肯定更省空间 
    if (filetype==TEXT) 
    {
        goToDelimiter(':', fi);
        for (a=0; a<layer1_size; a++) 
        {
            fscanf(fi, "%lf", &d);
            neu1[a].ac=d;
        }
    }
    if (filetype==BINARY) 
    {
        fgetc(fi);
        for (a=0; a<layer1_size; a++) 
        {
            fread(&fl, 4, 1, fi);
            neu1[a].ac=fl;
        }
    }
    //weight 0->1
    if (filetype==TEXT) 
    {
        goToDelimiter(':', fi);
        for (b=0; b<layer1_size; b++) 
        {
            for (a=0; a<layer0_size; a++) 
            {
                fscanf(fi, "%lf", &d);
                syn0[a+b*layer0_size].weight=d;
            }
        }
    }
    if (filetype==BINARY) 
    {
        for (b=0; b<layer1_size; b++) 
        {
            for (a=0; a<layer0_size; a++) 
            {
                fread(&fl, 4, 1, fi);
                syn0[a+b*layer0_size].weight=fl;
            }
        }
    }
    //weight 1->2
    if (filetype==TEXT) 
    {
        goToDelimiter(':', fi);
        if (layerc_size==0) 
        {	//no compress layer
            for (b=0; b<layer2_size; b++) 
            {
                for (a=0; a<layer1_size; a++) 
                {
                    fscanf(fi, "%lf", &d);
                    syn1[a+b*layer1_size].weight=d;
                }
            }
        }
        else
        {	//with compress layer
            for (b=0; b<layerc_size; b++) 
            {
                for (a=0; a<layer1_size; a++) 
                {
                    fscanf(fi, "%lf", &d);
                    syn1[a+b*layer1_size].weight=d;
                }
            }
            
            goToDelimiter(':', fi);
        
            for (b=0; b<layer2_size; b++)
            {
                for (a=0; a<layerc_size; a++) 
                {
                    fscanf(fi, "%lf", &d);
                    sync[a+b*layerc_size].weight=d;
                }
            }
        }
    }
    if (filetype==BINARY) 
    {
        if (layerc_size==0) 
        {	//no compress layer
            for (b=0; b<layer2_size; b++) 
            {
                for (a=0; a<layer1_size; a++) 
                {
                    fread(&fl, 4, 1, fi);
                    syn1[a+b*layer1_size].weight=fl;
                }
            }
        }
        else
        {				//with compress layer
            for (b=0; b<layerc_size; b++)
            {
                for (a=0; a<layer1_size; a++) 
                {
                    fread(&fl, 4, 1, fi);
                    syn1[a+b*layer1_size].weight=fl;
                }
            }
            
            for (b=0; b<layer2_size; b++) 
            {
                for (a=0; a<layerc_size; a++) 
                {
                    fread(&fl, 4, 1, fi);
                    sync[a+b*layerc_size].weight=fl;
                }
            }
        }
    }
    //
    if (filetype==TEXT) 
    {
        goToDelimiter(':', fi);		//direct conenctions
        long long aa;
        for (aa=0; aa<direct_size; aa++) 
        {
            fscanf(fi, "%lf", &d);
            syn_d[aa]=d;
        }
    }
    //
    if (filetype==BINARY) 
    {
        long long aa;
        for (aa=0; aa<direct_size; aa++)
        {
    	    fread(&fl, 4, 1, fi);
	        syn_d[aa]=fl;
	    
            /*fread(&si, 2, 1, fi);
            fl=si/(float)(4*256);
            syn_d[aa]=fl;*/
    	}
    }
    //
    
    saveWeights();

    fclose(fi);
}

//清除神经元的ac,er值  
void CRnnLM::netFlush()   //cleans all activations and error vectors
{
    int a;

    for (a=0; a<layer0_size-layer1_size; a++) {
        neu0[a].ac=0;
        neu0[a].er=0;
    }

    for (a=layer0_size-layer1_size; a<layer0_size; a++) {   //last hidden layer is initialized to vector of 0.1 values to prevent unstability
        neu0[a].ac=0.1;
        neu0[a].er=0;
    }

    for (a=0; a<layer1_size; a++) {
        neu1[a].ac=0;
        neu1[a].er=0;
    }
    
    for (a=0; a<layerc_size; a++) {
        neuc[a].ac=0;
        neuc[a].er=0;
    }
    
    for (a=0; a<layer2_size; a++) {
        neu2[a].ac=0;
        neu2[a].er=0;
    }
}

//将隐层神经元(论文中的状态层s(t))的ac值置1，s(t-1),即输入层layer1_size那部分的ac值置1，bptt+history清0
void CRnnLM::netReset()   //cleans hidden layer activation + bptt history
{
    int a, b;

    for (a=0; a<layer1_size; a++) 
    {
        neu1[a].ac=1.0;
    }

    copyHiddenLayerToInput();

    if (bptt>0) 
    {
        for (a=1; a<bptt+bptt_block; a++) 
            bptt_history[a]=0;
        for (a=bptt+bptt_block-1; a>1; a--) 
            for (b=0; b<layer1_size; b++) 
            {
                bptt_hidden[a*layer1_size+b].ac=0;
                bptt_hidden[a*layer1_size+b].er=0;
            }
    }

    for (a=0; a<MAX_NGRAM_ORDER; a++) 
        history[a]=0;
}

//matrixXvector(neu1, neu0, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 0);
//from:0  to:1
//from2:3 to2:5
//layer0_size:5
//layer1_size:2
//matrix_width:5
/*
下面这个函数用于权值矩阵乘以神经元向量,并将计算结果存入目的神经元向量，type == 0时,计算的是神经元ac值,
相当于计算srcmatrix × srcvec, 其中srcmatrix是(to-from)×(to2-from2)的矩阵，srcvec是(to2-from2)×1的
列向量,得到的结果是(to-from)×1的列向量,该列向量的值存入dest中的ac值；type == 1, 计算神经元的er值,即
(srcmatrix)^T × srcvec,T表示转置,转置后是(to2-from2)×(to-from),srcvec是(to-from)×1的列向量。这里的
矩阵相乘比下面被注释掉的的快,好像是叫做Strassen’s method，记不太清楚了,很久之前看算法导论时学的,感兴趣
的可以看看算法导论英文版第三版的79页，如果这不是Strassen’s method麻烦懂的朋友纠正一下~
*/
void CRnnLM::matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type)
{
    int a, b;
    real val1, val2, val3, val4;
    real val5, val6, val7, val8;
    
    if (type==0) 
    {//ac mod
        for (b=0; b<(to-from)/8; b++) 
        {
            val1=0;
            val2=0;
            val3=0;
            val4=0;
            
            val5=0;
            val6=0;
            val7=0;
            val8=0;
            
            for (a=from2; a<to2; a++) 
            {
                val1 += srcvec[a].ac * srcmatrix[a+(b*8+from+0)*matrix_width].weight;
                val2 += srcvec[a].ac * srcmatrix[a+(b*8+from+1)*matrix_width].weight;
                val3 += srcvec[a].ac * srcmatrix[a+(b*8+from+2)*matrix_width].weight;
                val4 += srcvec[a].ac * srcmatrix[a+(b*8+from+3)*matrix_width].weight;
                
                val5 += srcvec[a].ac * srcmatrix[a+(b*8+from+4)*matrix_width].weight;
                val6 += srcvec[a].ac * srcmatrix[a+(b*8+from+5)*matrix_width].weight;
                val7 += srcvec[a].ac * srcmatrix[a+(b*8+from+6)*matrix_width].weight;
                val8 += srcvec[a].ac * srcmatrix[a+(b*8+from+7)*matrix_width].weight;
            }
            dest[b*8+from+0].ac += val1;
            dest[b*8+from+1].ac += val2;
            dest[b*8+from+2].ac += val3;
            dest[b*8+from+3].ac += val4;
            
            dest[b*8+from+4].ac += val5;
            dest[b*8+from+5].ac += val6;
            dest[b*8+from+6].ac += val7;
            dest[b*8+from+7].ac += val8;
        }
        
        for (b=b*8; b<to-from; b++) 
        {
            for (a=from2; a<to2; a++) 
            {
                dest[b+from].ac += srcvec[a].ac * srcmatrix[a+(b+from)*matrix_width].weight;
            }
        }
    }
    else
    {		//er mod
    	for (a=0; a<(to2-from2)/8; a++) 
        {
            val1=0;
            val2=0;
            val3=0;
            val4=0;
            
            val5=0;
            val6=0;
            val7=0;
            val8=0;
            
            for (b=from; b<to; b++) 
            {
                val1 += srcvec[b].er * srcmatrix[a*8+from2+0+b*matrix_width].weight;
                val2 += srcvec[b].er * srcmatrix[a*8+from2+1+b*matrix_width].weight;
                val3 += srcvec[b].er * srcmatrix[a*8+from2+2+b*matrix_width].weight;
                val4 += srcvec[b].er * srcmatrix[a*8+from2+3+b*matrix_width].weight;
                
                val5 += srcvec[b].er * srcmatrix[a*8+from2+4+b*matrix_width].weight;
                val6 += srcvec[b].er * srcmatrix[a*8+from2+5+b*matrix_width].weight;
                val7 += srcvec[b].er * srcmatrix[a*8+from2+6+b*matrix_width].weight;
                val8 += srcvec[b].er * srcmatrix[a*8+from2+7+b*matrix_width].weight;
            }
            dest[a*8+from2+0].er += val1;
            dest[a*8+from2+1].er += val2;
            dest[a*8+from2+2].er += val3;
            dest[a*8+from2+3].er += val4;
            
            dest[a*8+from2+4].er += val5;
            dest[a*8+from2+5].er += val6;
            dest[a*8+from2+6].er += val7;
            dest[a*8+from2+7].er += val8;
	    }
	
	    for (a=a*8; a<to2-from2; a++) 
        {
            for (b=from; b<to; b++) 
            {
                dest[a+from2].er += srcvec[b].er * srcmatrix[a+from2+b*matrix_width].weight;
            }
    	}
    	
    	if (gradient_cutoff>0)
    	for (a=from2; a<to2; a++) 
        {
    	    if (dest[a].er>gradient_cutoff) dest[a].er=gradient_cutoff;
    	    if (dest[a].er<-gradient_cutoff) dest[a].er=-gradient_cutoff;
    	}
    }
    
    //this is normal implementation (about 3x slower):
    
    /*
    if (type==0) 
    {//ac mod
        for (b=from; b<to; b++) 
        {
            for (a=from2; a<to2; a++) 
            {
                dest[b].ac += srcvec[a].ac * srcmatrix[a+b*matrix_width].weight;
            }
        }
    }
    else 		//er mod
    if (type==1) 
    {
        for (a=from2; a<to2; a++) 
        {
            for (b=from; b<to; b++)
            {
    		    dest[a].er += srcvec[b].er * srcmatrix[a+b*matrix_width].weight;
    	    }
    	}
    }
    */
}





/*************************************************
 COMPUTE P(w|v,h)
 last_word表示当前输入层所在的词  
 word表示要预测的词 
*************************************************/

void CRnnLM::computeNet(int last_word, int word)
{
	computeClassProbs(last_word);
	if (gen>0) 
        return;	//if we generate words, we don't know what current word is -> only classes are estimated and word is selected in testGen()
	computeClassWordProbs(last_word, word);
}




/*************************************************
 COMPUTE CLASS PROBS
 计算P(Ci | s(t))
*************************************************/

void CRnnLM::computeClassProbs(int last_word)
{
    int a, b, c;
    real val;
    //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
    double sum;
    //real val1, val2, val3, val4;

    //将last_word对应的神经元ac值为1,也可以看做是对该词的1-of-V的编码 
    if (last_word!=-1) 
        neu0[last_word].ac=1;

    //propagate 0->1
    for (a=0; a<layer1_size; a++) 
        neu1[a].ac=0;
    for (a=0; a<layerc_size; a++) 
        neuc[a].ac=0;
    
    //这里计算的是s(t-1)与syn0的乘积
#ifdef USE_BLAS
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer1_size, layer1_size, 1.0, &syn0[layer0_size-layer1_size].weight,
    layer0_size, &neu0[layer0_size-layer1_size].ac, 2, 0.0, &neu1[0].ac, 2);
#else
    matrixXvector(neu1, neu0, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 0);
#endif

    //这里计算将last_word编码后的向量(大小是vocab_size,分量只有一个为1,其余为0)与syn0的乘积  
    for (b=0; b<layer1_size; b++) 
    {
        a=last_word;
        if (a!=-1) 
            neu1[b].ac += neu0[a].ac * syn0[a+b*layer0_size].weight;
    }

    //activate 1      --sigmoid
    //这里计算将上面隐层所得到的输入(ac值)经过sigmoid函数的映射结果 
    for (a=0; a<layer1_size; a++) 
    {
        //为数值稳定,将ac值大小限制在[-50,50]  
        //论文中有提到模型的参数小一些泛化的结果好一些 
	    if (neu1[a].ac>50) 
            neu1[a].ac=50;  //for numerical stability
        if (neu1[a].ac<-50) 
            neu1[a].ac=-50;  //for numerical stability
        val=-neu1[a].ac;
        //fasterexp函数在fasexp.h中实现,应该比math.h中的exp快吧  
        neu1[a].ac=1/(1+FAST_EXP(val)); //sigmoid函数即1/(1+e^(-x)) 
    }
    
    if (layerc_size>0) 
    {
        matrixXvector(neuc, neu1, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 0);
        //activate compression      --sigmoid
        for (a=0; a<layerc_size; a++) 
        {
            if (neuc[a].ac>50) 
                neuc[a].ac=50;  //for numerical stability
            if (neuc[a].ac<-50) 
                neuc[a].ac=-50;  //for numerical stability
            val=-neuc[a].ac;

            neuc[a].ac=1/(1+FAST_EXP(val));
        }
    }
        
    //1->2 class
    for (b=vocab_size; b<layer2_size; b++) 
        neu2[b].ac=0;
    
    if (layerc_size>0) 
    {
	    matrixXvector(neu2, neuc, sync, layerc_size, vocab_size, layer2_size, 0, layerc_size, 0);
    }
    else
    {
	    matrixXvector(neu2, neu1, syn1, layer1_size, vocab_size, layer2_size, 0, layer1_size, 0);
    }

    //apply direct connections to classes
    /*
     另外一个要说明的是最大熵模型，rnn结合了最大熵模型，直观的看上去是输入层与输出层连接了起来（虽然作者总是这么说，
     但我总觉的不能叫输入层和输出层连接起来，中间有过渡）。我们先看一下从神经网络的视角去看一个最大熵模型，这个神经
     网络就是没有隐层而已，其他和三层结构的一样，并且学习算法也是一样的。
    */
    if (direct_size>0) 
    {
        //注意这是hash定义在if内的,也就是出了if外面就无法访问了  
        //下面会看到每次都单独定义了局部的hash  
        //hash[i]里面存放的是i+1元模型的特征在syn_d中对应的下标 
        unsigned long long hash[MAX_NGRAM_ORDER];	//this will hold pointers to syn_d that contains hash parameters
        
        for (a=0; a<direct_order; a++) 
            hash[a]=0;
        
        //下面就是将n元特征单独映射为一个值,这里的权值是针对class部分的  
        for (a=0; a<direct_order; a++) 
        {
            b=0;
            if (a>0) if (history[a-1]==-1) break;	//if OOV was in history, do not use this N-gram feature and higher orders
            hash[a]=PRIMES[0]*PRIMES[1];
                    
            for (b=1; b<=a; b++) 
                hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);	//update hash value based on words from the history
            hash[a]=hash[a]%(direct_size/2);		//make sure that starting hash index is in the first half of syn_d (second part is reserved for history->words features)
        }

        /*
        我们把这段代码展开细走一下，假设direct_order = 3,并且没有OOV 
        out loop 1st: 
        a = 0; a < 4 
        b = 0; 
        hash[0]=PRIMES[0]*PRIMES[1] = 108641969 * 116049371; 
         
              inner loop 1st: 
              b = 1; b<=0 
              退出内循环 
              hash[0]=hash[0]%(direct_size/2) 
           
        out loop 2nd: 
        a = 1; a < 4; 
        b = 0; 
        hash[1]=PRIMES[0]*PRIMES[1] = 108641969 * 116049371; 
         
              inner loop 1st: 
              b = 1; b <= 1; 
              hash[a]= hash[a] + PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1) 
              = hash[1] + PRIMES[(1*PRIMES[1]+1)%36]*(history[0]+1); 
              退出内循环 
              hash[1]=hash[1]%(direct_size/2); 
               
        out loop 3rd: 
        a = 2; a < 4; 
        b = 0; 
        hash[2]=PRIMES[0]*PRIMES[1] = 108641969 * 116049371; 
         
              inner loop 1st: 
              b = 1; b <= 2; 
              hash[2]= hash[2] + PRIMES[(2*PRIMES[1]+1)%36]*(history[0]+1); 
               
              inner loop 2nd: 
              b = 2; b <= 2; 
              hash[2]= hash[2] + PRIMES[(2*PRIMES[2]+2)%PRIMES_SIZE]*(history[1]+1) 
              退出内循环 
                     
        大概能看出，hash[i]表示i+1元模型的历史映射，因为在计算hash[i]时，考虑了history[0..i-1] 
                      这个映射结果是作为syn_d数组的下标,i+1元词作为特征与输出层的连接真正的参数值在syn_d中 
        */    

        //ME部分,计算在class层的概率分布,即P(c i |s(t)) 
        //class_size = layer2_size - vocab_size 
        for (a=vocab_size; a<layer2_size; a++) 
        {
            for (b=0; b<direct_order; b++) 
                if (hash[b]) 
                {
                    neu2[a].ac+=syn_d[hash[b]];		//apply current parameter and move to the next one

                    //这里解释一下,i+1元特征与输出层所连接的参数是放在syn_d中  
                    //是连续的,这里连续的长度分两种情况,一种是对class计算的,有class_size的长度  
                    //另一种是对word的，连续的长度是word所对应类别的词数  
                    //后面类似的代码同理  
                    hash[b]++;
                } 
                else 
                    break;
        }
    }

    //activation 2   --softmax on classes
    //这里softmax归一概率  
    //这种方式主要是防止溢出,比如当ac值过大,exp(ac)可能就会溢出  
    sum=0;
    for (a=vocab_size; a<layer2_size; a++) 
    {
        if (neu2[a].ac>50) 
            neu2[a].ac=50;  //for numerical stability
        if (neu2[a].ac<-50) 
            neu2[a].ac=-50;  //for numerical stability
        val=FAST_EXP(neu2[a].ac);
        sum+=val;
        neu2[a].ac=val;
    }
    for (a=vocab_size; a<layer2_size; a++) 
        neu2[a].ac/=sum;         //output layer activations now sum exactly to 1
    
}




/*************************************************
 COMPUTE CLASS WORD PROBS
*************************************************/

void CRnnLM::computeClassWordProbs(int last_word, int word)
{
    int a, b, c;
    real val;
    double sum;   //sum is used for normalization: it's better to have larger precision as many numbers are summed together here
    real val1, val2, val3, val4;
   
    //1->2 word
    
    if (word!=-1) 
    {
        //class_cn[vocab[word].class_index]为某一class类中unique word个数
        for (c=0; c<class_cn[vocab[word].class_index]; c++) 
            neu2[class_words[vocab[word].class_index][c]].ac=0;
        if (layerc_size>0) 
        {
	        matrixXvector(neu2, neuc, sync, layerc_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layerc_size, 0);
        }
        else
        {
            matrixXvector(neu2, neu1, syn1, layer1_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layer1_size, 0);
        }
    }
    
    //apply direct connections to words
    if (word!=-1) if (direct_size>0) 
    {
        unsigned long long hash[MAX_NGRAM_ORDER];
            
        for (a=0; a<direct_order; a++) 
            hash[a]=0;
        
        for (a=0; a<direct_order; a++)
        {
            b=0;
            if (a>0) if (history[a-1]==-1) break;
            hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(vocab[word].class_index+1);
                    
            for (b=1; b<=a; b++) 
                hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
            hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
        }
        
        for (c=0; c<class_cn[vocab[word].class_index]; c++) 
        {
            a=class_words[vocab[word].class_index][c];
            
            for (b=0; b<direct_order; b++) 
                if (hash[b]) 
                {
                    neu2[a].ac+=syn_d[hash[b]];
                    hash[b]++;
                    hash[b]=hash[b]%direct_size;
                }
                else 
                    break;
        }
    }

    //activation 2   --softmax on words
    sum=0;
    if (word!=-1) 
    {
        for (c=0; c<class_cn[vocab[word].class_index]; c++) 
        {
            a=class_words[vocab[word].class_index][c];
            if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
            if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
            
            val=FAST_EXP(neu2[a].ac);
            sum+=val;
            neu2[a].ac=val;
        }
        for (c=0; c<class_cn[vocab[word].class_index]; c++) 
            neu2[class_words[vocab[word].class_index][c]].ac/=sum;
    }
}

//word表示要预测的词,last_word表示当前输入层所在的词
void CRnnLM::learnNet(int last_word, int word)
{
    int a, b, c, t, step;
    real beta2, beta3;

    //alpha表示学习率,初始值为0.1, beta初始值为0.0000001; 
    beta2=beta*alpha;
    beta3=beta2*1;	//beta3 can be possibly larger than beta2, as that is useful on small datasets (if the final model is to be interpolated wich backoff model) - TODO in the future

    if (word==-1) 
        return;

    //compute error vectors，计算输出层的(只含word所在类别的所有词)误差向量  
    for (c=0; c<class_cn[vocab[word].class_index]; c++) 
    {
	    a=class_words[vocab[word].class_index][c];
        neu2[a].er=(0-neu2[a].ac); //class所含的word中，其它维度的标签都为0，只有word所对应的维度为1，详情请看word part
    }
    neu2[word].er=(1-neu2[word].ac);	//word part

    //flush error
    for (a=0; a<layer1_size; a++) neu1[a].er=0;
    for (a=0; a<layerc_size; a++) neuc[a].er=0;

    //计算输出层的class部分的误差向量  
    for (a=vocab_size; a<layer2_size; a++) 
    {
        neu2[a].er=(0-neu2[a].ac);
    }
    neu2[vocab[word].class_index+vocab_size].er=(1-neu2[vocab[word].class_index+vocab_size].ac);	//class part
    
    //计算特征所在syn_d中的下标，和上面一样，针对ME中word部分  
    if (direct_size>0) 
    {	//learn direct connections between words
        if (word!=-1) 
        {
            unsigned long long hash[MAX_NGRAM_ORDER];
            
            for (a=0; a<direct_order; a++) hash[a]=0;
        
            for (a=0; a<direct_order; a++) 
            {
                b=0;
                if (a>0) if (history[a-1]==-1) break;
                hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(vocab[word].class_index+1);
                        
                for (b=1; b<=a; b++) 
                    hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
                hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
            }
        
            //更新ME中的权值部分，这部分是正对word的  
            for (c=0; c<class_cn[vocab[word].class_index]; c++) 
            {
                a=class_words[vocab[word].class_index][c];
                
                for (b=0; b<direct_order; b++) 
                    if (hash[b]) 
                    {
                        syn_d[hash[b]]+=alpha*neu2[a].er - syn_d[hash[b]]*beta3;
                        hash[b]++;
                        hash[b]=hash[b]%direct_size;
                    } 
                    else 
                        break;
            }
        }
    }
    //计算n元模型特征,这是对class计算的 
    //learn direct connections to classes
    if (direct_size>0) 
    {	//learn direct connections between words and classes
        unsigned long long hash[MAX_NGRAM_ORDER];
        
        for (a=0; a<direct_order; a++) hash[a]=0;
        
        for (a=0; a<direct_order; a++) 
        {
            b=0;
            if (a>0) if (history[a-1]==-1) break;
            hash[a]=PRIMES[0]*PRIMES[1];
                    
            for (b=1; b<=a; b++) 
                hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
            hash[a]=hash[a]%(direct_size/2);
        }
        
        for (a=vocab_size; a<layer2_size; a++) 
        {
            for (b=0; b<direct_order; b++) 
                if (hash[b]) 
                {
                    syn_d[hash[b]]+=alpha*neu2[a].er - syn_d[hash[b]]*beta3;
                    hash[b]++;
                } 
                else 
                    break;
        }
    }
    //
    
    //含压缩层的情况，更新sync, syn1 
    if (layerc_size>0) 
    {
        matrixXvector(neuc, neu2, sync, layerc_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layerc_size, 1);
        
        t=class_words[vocab[word].class_index][0]*layerc_size;
        for (c=0; c<class_cn[vocab[word].class_index]; c++) 
        {
            b=class_words[vocab[word].class_index][c];
            if ((counter%10)==0)	//regularization is done every 10. step
                for (a=0; a<layerc_size; a++) 
                    sync[a+t].weight+=alpha*neu2[b].er*neuc[a].ac - sync[a+t].weight*beta2;
            else
                for (a=0; a<layerc_size; a++) 
                    sync[a+t].weight+=alpha*neu2[b].er*neuc[a].ac;
            t+=layerc_size;
        }
        //
        matrixXvector(neuc, neu2, sync, layerc_size, vocab_size, layer2_size, 0, layerc_size, 1);		//propagates errors 2->c for classes
        
        c=vocab_size*layerc_size;
        for (b=vocab_size; b<layer2_size; b++) 
        {
            if ((counter%10)==0) 
            {	//regularization is done every 10. step
                for (a=0; a<layerc_size; a++) 
                    sync[a+c].weight+=alpha*neu2[b].er*neuc[a].ac - sync[a+c].weight*beta2;	//weight c->2 update
            }
            else 
            {
                for (a=0; a<layerc_size; a++) 
                    sync[a+c].weight+=alpha*neu2[b].er*neuc[a].ac;	//weight c->2 update
            }
            c+=layerc_size;
        }
        
        for (a=0; a<layerc_size; a++) 
            neuc[a].er=neuc[a].er*neuc[a].ac*(1-neuc[a].ac);    //error derivation at compression layer

        ////
        
        matrixXvector(neu1, neuc, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 1);		//propagates errors c->1
        
        for (b=0; b<layerc_size; b++) 
        {
            for (a=0; a<layer1_size; a++) 
                syn1[a+b*layer1_size].weight+=alpha*neuc[b].er*neu1[a].ac;	//weight 1->c update
        }
    }
    else
    {
    	matrixXvector(neu1, neu2, syn1, layer1_size, class_words[vocab[word].class_index][0], class_words[vocab[word].class_index][0]+class_cn[vocab[word].class_index], 0, layer1_size, 1);
    	
    	t=class_words[vocab[word].class_index][0]*layer1_size;
	    for (c=0; c<class_cn[vocab[word].class_index]; c++) 
        {
            b=class_words[vocab[word].class_index][c];
            if ((counter%10)==0)	//regularization is done every 10. step
                for (a=0; a<layer1_size; a++) 
                    syn1[a+t].weight+=alpha*neu2[b].er*neu1[a].ac - syn1[a+t].weight*beta2;
            else
                for (a=0; a<layer1_size; a++) 
                    syn1[a+t].weight+=alpha*neu2[b].er*neu1[a].ac;
            t+=layer1_size;
        }
        //
        matrixXvector(neu1, neu2, syn1, layer1_size, vocab_size, layer2_size, 0, layer1_size, 1);		//propagates errors 2->1 for classes
        
        c=vocab_size*layer1_size;
        for (b=vocab_size; b<layer2_size; b++) 
        {
            if ((counter%10)==0) 
            {	//regularization is done every 10. step
                for (a=0; a<layer1_size; a++) 
                    syn1[a+c].weight+=alpha*neu2[b].er*neu1[a].ac - syn1[a+c].weight*beta2;	//weight 1->2 update
            }
            else 
            {
                for (a=0; a<layer1_size; a++) 
                    syn1[a+c].weight+=alpha*neu2[b].er*neu1[a].ac;	//weight 1->2 update
            }
            c+=layer1_size;
        }
    }
    
    //
    
    ///////////////

    if (bptt<=1) 
    {		//bptt==1 -> normal BP
        for (a=0; a<layer1_size; a++) neu1[a].er=neu1[a].er*neu1[a].ac*(1-neu1[a].ac);    //error derivation at layer 1

        //weight update 1->0
        a=last_word;
        if (a!=-1) {
            if ((counter%10)==0)
            for (b=0; b<layer1_size; b++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac - syn0[a+b*layer0_size].weight*beta2;
            else
            for (b=0; b<layer1_size; b++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac;
        }

        if ((counter%10)==0) {
            for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac - syn0[a+b*layer0_size].weight*beta2;
        }
        else {
            for (b=0; b<layer1_size; b++) for (a=layer0_size-layer1_size; a<layer0_size; a++) syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac;
        }
    }
    else		//BPTT
    {
        for (b=0; b<layer1_size; b++) bptt_hidden[b].ac=neu1[b].ac;
        for (b=0; b<layer1_size; b++) bptt_hidden[b].er=neu1[b].er;
        
        if (((counter%bptt_block)==0) || (independent && (word==0))) 
        {
            for (step=0; step<bptt+bptt_block-2; step++) 
            {
                for (a=0; a<layer1_size; a++) 
                    neu1[a].er=neu1[a].er*neu1[a].ac*(1-neu1[a].ac);    //error derivation at layer 1

                //weight update 1->0
                a=bptt_history[step];
                if (a!=-1)
                for (b=0; b<layer1_size; b++) 
                {
                        bptt_syn0[a+b*layer0_size].weight+=alpha*neu1[b].er;//*neu0[a].ac; --should be always set to 1
                }
                
                for (a=layer0_size-layer1_size; a<layer0_size; a++) 
                    neu0[a].er=0;
                
                matrixXvector(neu0, neu1, syn0, layer0_size, 0, layer1_size, layer0_size-layer1_size, layer0_size, 1);		//propagates errors 1->0
                for (b=0; b<layer1_size; b++) 
                    for (a=layer0_size-layer1_size; a<layer0_size; a++) 
                    {
                        //neu0[a].er += neu1[b].er * syn0[a+b*layer0_size].weight;
                        bptt_syn0[a+b*layer0_size].weight+=alpha*neu1[b].er*neu0[a].ac;
                    }
                
                for (a=0; a<layer1_size; a++) 
                {	//propagate error from time T-n to T-n-1
                    neu1[a].er=neu0[a+layer0_size-layer1_size].er + bptt_hidden[(step+1)*layer1_size+a].er;
                }
                
                if (step<bptt+bptt_block-3)
                    for (a=0; a<layer1_size; a++)
                    {
                        neu1[a].ac=bptt_hidden[(step+1)*layer1_size+a].ac;
                        neu0[a+layer0_size-layer1_size].ac=bptt_hidden[(step+2)*layer1_size+a].ac;
                    }
            }
            
            for (a=0; a<(bptt+bptt_block)*layer1_size; a++) 
            {
                bptt_hidden[a].er=0;
            }
        
        
            for (b=0; b<layer1_size; b++) 
                neu1[b].ac=bptt_hidden[b].ac;		//restore hidden layer after bptt
            
        
            //
            for (b=0; b<layer1_size; b++) 
            {		//copy temporary syn0
                if ((counter%10)==0) 
                {
                    for (a=layer0_size-layer1_size; a<layer0_size; a++) 
                    {
                        syn0[a+b*layer0_size].weight+=bptt_syn0[a+b*layer0_size].weight - syn0[a+b*layer0_size].weight*beta2;
                        bptt_syn0[a+b*layer0_size].weight=0;
                    }
                }
                else 
                {
                    for (a=layer0_size-layer1_size; a<layer0_size; a++) 
                    {
                        syn0[a+b*layer0_size].weight+=bptt_syn0[a+b*layer0_size].weight;
                        bptt_syn0[a+b*layer0_size].weight=0;
                    }
                }
                
                if ((counter%10)==0) 
                {
                    for (step=0; step<bptt+bptt_block-2; step++) 
                        if (bptt_history[step]!=-1) 
                        {
                            syn0[bptt_history[step]+b*layer0_size].weight+=bptt_syn0[bptt_history[step]+b*layer0_size].weight - syn0[bptt_history[step]+b*layer0_size].weight*beta2;
                            bptt_syn0[bptt_history[step]+b*layer0_size].weight=0;
                        }
                }
                else 
                {
                    for (step=0; step<bptt+bptt_block-2; step++) 
                        if (bptt_history[step]!=-1) 
                        {
                            syn0[bptt_history[step]+b*layer0_size].weight+=bptt_syn0[bptt_history[step]+b*layer0_size].weight;
                            bptt_syn0[bptt_history[step]+b*layer0_size].weight=0;
                        }
                }
            }
        }
    }	
}





// 
// 
// #define BOUND_A 0.5
// #define VALUE_A 0.00
// #define BOUND_B 0.3
// #define VALUE_B 0.2
// #define BOUND_C 0.7
// #define VALUE_C 0.5
// #define VALUE_OTHER 1.0








void CRnnLM::copyHiddenLayerToInput()
{
    int a;
    //real tab[100] = {0.166055, 0.029667, 0.142579, 0.199170, 0.070865, 0.067237, 0.183381, 0.059361, 0.120684, 0.078472, 0.051646, 0.107730, 0.136236, 0.055967, 0.088278, 0.178920, 0.080591, 0.199652, 0.033339, 0.085404, 0.145996, 0.036254, 0.086556, 0.099282, 0.125244, 0.103773, 0.155118, 0.138335, 0.076856, 0.045497, 0.061591, 0.060743, 0.042819, 0.098242, 0.065803, 0.072250, 0.102154, 0.238162, 0.092465, 0.101421, 0.050651, 0.072009, 0.037655, 0.141597, 0.261121, 0.188973, 0.058801, 0.078184, 0.090921, 0.131005, 0.074584, 0.094283, 0.092171, 0.203648, 0.082538, 0.159812, 0.122438, 0.134544, 0.049062, 0.118686, 0.061725, 0.107930, 0.064489, 0.148496, 0.109443, 0.086788, 0.098533, 0.104770, 0.111259, 0.052291, 0.053542, 0.165732, 0.141032, 0.068891, 0.079476, 0.039536, 0.089736, 0.171921, 0.088489, 0.293489, 0.182057, 0.089817, 0.051818, 0.177904, 0.254662, 0.173513, 0.205199, 0.060849, 0.262845, 0.077591, 0.037978, 0.125393, 0.068918, 0.066538, 0.080752, 0.127415, 0.109560, 0.039830, 0.067095, 0.100014};
    //real tab[300] = {0.020, 0.054, 0.088, 0.025, 0.003, 0.033, 0.002, 0.042, 0.050, 0.002, 0.120, 0.002, 0.042, 0.069, 0.107, 0.027, 0.088, 0.005, 0.002, 0.065, 0.085, 0.045, 0.025, 0.020, 0.039, 0.048, 0.094, 0.096, 0.149, 0.114, 0.045, 0.081, 0.003, 0.003, 0.035, 0.068, 0.191, 0.015, 0.047, 0.001, 0.002, 0.073, 0.002, 0.103, 0.094, 0.027, 0.001, 0.003, 0.066, 0.044, 0.141, 0.068, 0.105, 0.002, 0.001, 0.014, 0.094, 0.064, 0.002, 0.036, 0.003, 0.003, 0.110, 0.148, 0.049, 0.025, 0.224, 0.198, 0.001, 0.078, 0.095, 0.001, 0.168, 0.001, 0.002, 0.066, 0.064, 0.088, 0.001, 0.106, 0.001, 0.047, 0.128, 0.025, 0.073, 0.054, 0.161, 0.179, 0.057, 0.001, 0.036, 0.056, 0.056, 0.069, 0.003, 0.043, 0.086, 0.045, 0.002, 0.003, 0.219, 0.031, 0.116, 0.001, 0.013, 0.124, 0.033, 0.047, 0.016, 0.001, 0.002, 0.028, 0.028, 0.001, 0.002, 0.053, 0.008, 0.138, 0.001, 0.002, 0.190, 0.059, 0.032, 0.054, 0.001, 0.138, 0.001, 0.006, 0.073, 0.002, 0.002, 0.012, 0.061, 0.111, 0.077, 0.206, 0.099, 0.122, 0.136, 0.083, 0.019, 0.031, 0.110, 0.001, 0.053, 0.058, 0.079, 0.033, 0.169, 0.034, 0.266, 0.001, 0.064, 0.261, 0.001, 0.005, 0.144, 0.074, 0.013, 0.079, 0.031, 0.071, 0.011, 0.067, 0.002, 0.020, 0.001, 0.001, 0.142, 0.074, 0.030, 0.001, 0.084, 0.002, 0.024, 0.022, 0.048, 0.059, 0.135, 0.119, 0.144, 0.001, 0.078, 0.023, 0.049, 0.014, 0.001, 0.085, 0.030, 0.101, 0.004, 0.030, 0.003, 0.006, 0.060, 0.032, 0.131, 0.090, 0.013, 0.081, 0.130, 0.001, 0.001, 0.023, 0.001, 0.019, 0.018, 0.128, 0.002, 0.046, 0.005, 0.017, 0.084, 0.071, 0.032, 0.001, 0.103, 0.003, 0.001, 0.001, 0.004, 0.047, 0.200, 0.028, 0.005, 0.139, 0.007, 0.038, 0.010, 0.051, 0.108, 0.074, 0.038, 0.001, 0.001, 0.130, 0.001, 0.077, 0.001, 0.046, 0.090, 0.002, 0.034, 0.121, 0.001, 0.261, 0.001, 0.041, 0.001, 0.064, 0.001, 0.043, 0.152, 0.182, 0.022, 0.001, 0.001, 0.197, 0.020, 0.034, 0.134, 0.022, 0.003, 0.002, 0.018, 0.026, 0.003, 0.023, 0.010, 0.047, 0.029, 0.035, 0.073, 0.226, 0.223, 0.033, 0.032, 0.001, 0.074, 0.073, 0.011, 0.083, 0.059, 0.165, 0.086, 0.004, 0.172, 0.002, 0.082, 0.004, 0.048, 0.036, 0.034, 0.035, 0.087, 0.002, 0.160, 0.062, 0.083, 0.086};
    //float v = 0.0;
	if (disc_map_set == 1) 
    {
		HierarchicalClusterFstHistory *p = dynamic_cast<HierarchicalClusterFstHistory *>(fsth);
		fsth->setFstHistory(*this, *d);
		fsth->loadAsInput(*this, *d);
        //for (a=0; a<layer1_size; a++) 
        //{
        //  printf(" %.3f", neu0[a+layer0_size-layer1_size].ac);
        //}
        //printf("\n");
	}
	else 
    {
		for (a=0; a<layer1_size; a++) 
        {
            //v = neu1[a].ac;
            // if (v...
		    //neu0[a+layer0_size-layer1_size].ac=tab[a];
		    neu0[a+layer0_size-layer1_size].ac=neu1[a].ac;
	    }
	}
}

void CRnnLM::trainNet()
{
    int a, b, word, last_word, wordcn;
    char log_name[200];
    FILE *fi, *flog;
    clock_t start, now;

    sprintf(log_name, "%s.output.txt", rnnlm_file);

    printf("Starting training using file %s\n", train_file);
    starting_alpha=alpha;
    
    fi=fopen(rnnlm_file, "rb");
    if (fi!=NULL) 
    {
        fclose(fi);
        printf("Restoring network from file to continue training...\n");
        restoreNet();
    } 
    else 
    {
        learnVocabFromTrainFile();
        initNet();
        iter=0;
    }

    if (class_size>vocab_size) 
    {
	    printf("WARNING: number of classes exceeds vocabulary size!\n");
    }
    
    counter=train_cur_pos;
    
    //saveNet();

    while (1) 
    {
        printf("Iter: %3d\tAlpha: %f\t   ", iter, alpha);
        fflush(stdout);
        
        if (bptt>0) 
            for (a=0; a<bptt+bptt_block; a++) 
                bptt_history[a]=0;
        for (a=0; a<MAX_NGRAM_ORDER; a++) 
            history[a]=0;

        //TRAINING PHASE
        netFlush();

        fi=fopen(train_file, "rb");
        last_word=0;
        
        if (counter>0) 
            for (a=0; a<counter; a++) 
                word=readWordIndex(fi);	//this will skip words that were already learned if the training was interrupted
        
        start=clock();
        
        while (1) 
        {
    	    counter++;
    	    
    	    if ((counter%10000)==0) if ((debug_mode>1)) 
            {
                now=clock();
                if (train_words>0)
                    printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f ", 13, iter, alpha, -logp/log10(2)/counter, counter/(real)train_words*100, counter/((double)(now-start)/1000000.0));
                else
                    printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %dK", 13, iter, alpha, -logp/log10(2)/counter, counter/1000);
                fflush(stdout);
    	    }
    	    
    	    if ((anti_k>0) && ((counter%anti_k)==0)) 
            {
                train_cur_pos=counter;
                saveNet();
    	    }
        
	        word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) 
                break;        //end of file: test on validation data, iterate till convergence

            if (word!=-1) 
                logp+=log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
    	    
    	    if ((logp!=logp) || (isinf(logp))) 
            {
    	        printf("\nNumerical error %d %f %f\n", word, neu2[word].ac, neu2[vocab[word].class_index+vocab_size].ac);
    	        exit(1);
    	    }
	    
            //
            if (bptt>0) 
            {		//shift memory needed for bptt to next time step
                for (a=bptt+bptt_block-1; a>0; a--) 
                    bptt_history[a]=bptt_history[a-1];
                bptt_history[0]=last_word;
                
                for (a=bptt+bptt_block-1; a>0; a--) 
                    for (b=0; b<layer1_size; b++) 
                    {
                        bptt_hidden[a*layer1_size+b].ac=bptt_hidden[(a-1)*layer1_size+b].ac;
                        bptt_hidden[a*layer1_size+b].er=bptt_hidden[(a-1)*layer1_size+b].er;
                    }
            }
            //
            learnNet(last_word, word);
            
            copyHiddenLayerToInput();

            if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

            last_word=word;
            
            for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
            history[0]=last_word;

	        if (independent && (word==0)) 
                netReset();
        }
        fclose(fi);

	    now=clock();
    	printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 13, iter, alpha, -logp/log10(2)/counter, counter/((double)(now-start)/1000000.0));
   
    	if (one_iter==1) 
        {	//no validation data are needed and network is always saved with modified weights
    	    printf("\n");
	        logp=0;
    	    saveNet();
            break;
    	}

        //VALIDATION PHASE
        netFlush();

        fi=fopen(valid_file, "rb");
	    if (fi==NULL) 
        {
            printf("Valid file not found\n");
            exit(1);
	    }
        
        flog=fopen(log_name, "ab");
	    if (flog==NULL)     
        {
            printf("Cannot open log file\n");
            exit(1);
	    }
        
        //fprintf(flog, "Index   P(NET)          Word\n");
        //fprintf(flog, "----------------------------------\n");
        
        last_word=0;
        logp=0;
        wordcn=0;
        while (1) 
        {
            word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) break;        //end of file: report LOGP, PPL
            
    	    if (word!=-1) 
            {
                logp+=log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
                wordcn++;
    	    }

            /*if (word!=-1)
                fprintf(flog, "%d\t%f\t%s\n", word, neu2[word].ac, vocab[word].word);
            else
                fprintf(flog, "-1\t0\t\tOOV\n");*/

            //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
            copyHiddenLayerToInput();

            if (last_word!=-1) 
                neu0[last_word].ac=0;  //delete previous activation

            last_word=word;
            
            for (a=MAX_NGRAM_ORDER-1; a>0; a--) 
                history[a]=history[a-1];
            history[0]=last_word;

	        if (independent && (word==0)) 
                netReset();
        }
        fclose(fi);
        
        fprintf(flog, "\niter: %d\n", iter);
        fprintf(flog, "valid log probability: %f\n", logp);
        fprintf(flog, "PPL net: %f\n", exp10(-logp/(real)wordcn));
        
        fclose(flog);
    
        printf("VALID entropy: %.4f\n", -logp/log10(2)/wordcn);
        
        counter=0;
	    train_cur_pos=0;

        if (logp<llogp)
            restoreWeights();
        else
            saveWeights();

        if (logp*min_improvement<llogp) 
        {
            if (alpha_divide==0) 
                alpha_divide=1;
            else 
            {
                saveNet();
                break;
            }
        }

        if (alpha_divide) alpha/=2;

        llogp=logp;
        logp=0;
        iter++;
        saveNet();
    }
}

void CRnnLM::testNet()
{
    int a, b, i, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob;
    char str[MAX_STRING];
    real prob_other, log_other, log_combine, f;
    
    restoreNet();
    
    if (use_lmprob) {
	lmprob=fopen(lmprob_file, "rb");
    }

    //TEST PHASE
    //netFlush();

    fi=fopen(test_file, "rb");
    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
    //flog=fopen(str, "wb");
    flog=stdout;

    if (debug_mode>1)	{
	if (use_lmprob) {
    	    fprintf(flog, "Index   P(NET)          P(LM)           Word\n");
    	    fprintf(flog, "--------------------------------------------------\n");
	} else {
    	    fprintf(flog, "Index   P(NET)          Word\n");
    	    fprintf(flog, "----------------------------------\n");
	}
    }

    last_word=0;					//last word = end of sentence
    logp=0;
    log_other=0;
    log_combine=0;
    prob_other=0;
    wordcn=0;
    copyHiddenLayerToInput();
    int utt_nw = 0;
    real utt_logp =0.0;
    
    
    if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=-1;
    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
    if (independent) netReset();
    
    while (1) {

        word=readWordIndex(fi);		//read next word 
        computeNet(last_word, word);		//compute probability distribution
        
        if (feof(fi)) break;		//end of file: report LOGP, PPL
        
        if (use_lmprob) {
    	    if (sizeof(real)>4)
        	fscanf(lmprob, "%lf", &prob_other);
    	    else
    		fscanf(lmprob, "%f", &prob_other);
    		
            goToDelimiter('\n', lmprob);
        }

        if ((word!=-1) || (prob_other>0)) {
    	    if (word==-1) {
    		logp+=-8;		//some ad hoc penalty - when mixing different vocabularies, single model score is not real PPL
        	log_combine+=log10(0 * lambda + prob_other*(1-lambda));
    	    } else {
    		logp+=log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
        	log_combine+=log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac*lambda + prob_other*(1-lambda));
    	    }
    	    log_other+=log10(prob_other);
            wordcn++;
        }

	if (debug_mode>1) {
    	    if (use_lmprob) {
        	if (word!=-1) {
        	    fprintf(flog, "%d\t%.10f\t%.10f\t%s", word, neu2[vocab[word].class_index+vocab_size].ac *neu2[word].ac, prob_other, vocab[word].word);
    	            utt_logp += log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac*lambda + prob_other*(1-lambda));
    	            utt_nw++;
    	        }
        	else fprintf(flog, "-1\t0\t\t0\t\tOOV");
    	    } else {
        	if (word!=-1) {
        	    fprintf(flog, "%d\t%.10f\t%s", word, neu2[vocab[word].class_index+vocab_size].ac *neu2[word].ac, vocab[word].word);
    	            utt_logp += log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
    	            utt_nw++;
    	        }
        	else fprintf(flog, "-1\t0\t\tOOV");
    	    }
    	    
    	    fprintf(flog,"\n");
    	    

    	    
    	    
        	if (word == 0) {
		printf("\n------------------------------------------------------------------------\n");
		printf("LogP = %f \tLogP (base 10) = %f \tPPL = %f\n", utt_logp/log10(exp(1)), utt_logp, exp10(-utt_logp/(float) utt_nw));
		printf("------------------------------------------------------------------------\n");
		utt_logp =0.0;
		utt_nw=0;
		}
    	}

        if (dynamic>0) {
            if (bptt>0) {
                for (a=bptt+bptt_block-1; a>0; a--) bptt_history[a]=bptt_history[a-1];
                bptt_history[0]=last_word;
                                    
                for (a=bptt+bptt_block-1; a>0; a--) for (b=0; b<layer1_size; b++) {
                    bptt_hidden[a*layer1_size+b].ac=bptt_hidden[(a-1)*layer1_size+b].ac;
                    bptt_hidden[a*layer1_size+b].er=bptt_hidden[(a-1)*layer1_size+b].er;
        	}
            }
            //
            alpha=dynamic;
    	    learnNet(last_word, word);    //dynamic update
    	}
        copyHiddenLayerToInput();
        
        if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation
        last_word=word;
	
	for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
	history[0]=last_word;
	
	if (independent && (word==0)) netReset();
    }
    fclose(fi);
    if (use_lmprob) fclose(lmprob);

    //write to log file
    if (debug_mode>0) {
	fprintf(flog, "\ntest log probability: %f\n", logp);
	if (use_lmprob) {
    	    fprintf(flog, "test log probability given by other lm: %f\n", log_other);
    	    fprintf(flog, "test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
	}

	fprintf(flog, "\nPPL net: %f\n", exp10(-logp/(real)wordcn));
	if (use_lmprob) {
    	    fprintf(flog, "PPL other: %f\n", exp10(-log_other/(real)wordcn));
    	    fprintf(flog, "PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
	}
    }
    
    fclose(flog);
}

void CRnnLM::testNbest()
{
    int a, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob;
    char str[MAX_STRING];
    float prob_other; //has to be float so that %f works in fscanf
    real log_other, log_combine, senp;
    //int nbest=-1;
    int nbest_cn=0;
    char ut1[MAX_STRING], ut2[MAX_STRING];

    restoreNet();
    computeNet(0, 0);
    copyHiddenLayerToInput();
    saveContext();
    saveContext2();
    
    if (use_lmprob) {
	lmprob=fopen(lmprob_file, "rb");
    } else lambda=1;		//!!! for simpler implementation later

    //TEST PHASE
    //netFlush();

    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
    
    if (!strcmp(test_file, "-")) fi=stdin; else fi=fopen(test_file, "rb");
    
    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
    //flog=fopen(str, "wb");
    flog=stdout;

    last_word=0;		//last word = end of sentence
    logp=0;
    log_other=0;
    prob_other=0;
    log_combine=0;
    wordcn=0;
    senp=0;
    strcpy(ut1, (char *)"");
    while (1) {
	if (last_word==0) {
	    fscanf(fi, "%s", ut2);
	    
	    if (nbest_cn==1) saveContext2();		//save context after processing first sentence in nbest
	    
	    if (strcmp(ut1, ut2)) {
		strcpy(ut1, ut2);
		nbest_cn=0;
		restoreContext2();
		saveContext();
	    } else restoreContext();
	    
	    nbest_cn++;
	    
	    copyHiddenLayerToInput();
        }
    
	
	word=readWordIndex(fi);     //read next word
	if (lambda>0) computeNet(last_word, word);      //compute probability distribution
        if (feof(fi)) break;        //end of file: report LOGP, PPL
        
        
        if (use_lmprob) {
            fscanf(lmprob, "%f", &prob_other);
            goToDelimiter('\n', lmprob);
        }
        
        if (word!=-1)
        neu2[word].ac*=neu2[vocab[word].class_index+vocab_size].ac;
        
        if (word!=-1) {
            logp+=log10(neu2[word].ac);
    	    
            log_other+=log10(prob_other);
            
            log_combine+=log10(neu2[word].ac*lambda + prob_other*(1-lambda));
            
            senp+=log10(neu2[word].ac*lambda + prob_other*(1-lambda));
            
            wordcn++;
        } else {
    	    //assign to OOVs some score to correctly rescore nbest lists, reasonable value can be less than 1/|V| or backoff LM score (in case it is trained on more data)
    	    //this means that PPL results from nbest list rescoring are not true probabilities anymore (as in open vocabulary LMs)
    	    
    	    real oov_penalty=-5;	//log penalty
    	    
    	    if (prob_other!=0) {
    		logp+=log10(prob_other);
    		log_other+=log10(prob_other);
    		log_combine+=log10(prob_other);
    		senp+=log10(prob_other);
    	    } else {
    		logp+=oov_penalty;
    		log_other+=oov_penalty;
    		log_combine+=oov_penalty;
    		senp+=oov_penalty;
    	    }
    	    wordcn++;
        }
        
        //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
        copyHiddenLayerToInput();

        if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation
        
        if (word==0) {		//write last sentence log probability / likelihood
    	    fprintf(flog, "%f\n", senp);
    	    senp=0;
	}

        last_word=word;
	
	for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
	history[0]=last_word;
	
	if (independent && (word==0)) netReset();
    }
    fclose(fi);
    if (use_lmprob) fclose(lmprob);

    if (debug_mode>0) {
	printf("\ntest log probability: %f\n", logp);
	if (use_lmprob) {
    	    printf("test log probability given by other lm: %f\n", log_other);
    	    printf("test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
	}

	printf("\nPPL net: %f\n", exp10(-logp/(real)wordcn));
	if (use_lmprob) {
    	    printf("PPL other: %f\n", exp10(-log_other/(real)wordcn));
    	    printf("PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
	}
    }

    fclose(flog);
}

void CRnnLM::testGen()
{
    int i, word, cla, last_word, wordcn, c, b, a=0;
    real f, g, sum, val;
    
    restoreNet();
    
    word=0;
    last_word=0;					//last word = end of sentence
    wordcn=0;
    copyHiddenLayerToInput();
    while (wordcn<gen) {
        computeClassProbs(last_word);		//compute probability distribution
        
        f=random(0, 1);
        g=0;
        i=vocab_size;
        while ((g<f) && (i<layer2_size)) {
    	    g+=neu2[i].ac;
    	    i++;
        }
        cla=i-1-vocab_size;
        
        if (cla>class_size-1) cla=class_size-1;
        if (cla<0) cla=0;
        
        //
        // !!!!!!!!  THIS WILL WORK ONLY IF CLASSES ARE CONTINUALLY DEFINED IN VOCAB !!! (like class 10 = words 11 12 13; not 11 12 16)  !!!!!!!!
        // forward pass 1->2 for words
        for (c=0; c<class_cn[cla]; c++) neu2[class_words[cla][c]].ac=0;
        matrixXvector(neu2, neu1, syn1, layer1_size, class_words[cla][0], class_words[cla][0]+class_cn[cla], 0, layer1_size, 0);
	
	//apply direct connections to words
	if (word!=-1) if (direct_size>0) {
    	    unsigned long long hash[MAX_NGRAM_ORDER];

            for (a=0; a<direct_order; a++) hash[a]=0;

            for (a=0; a<direct_order; a++) {
                b=0;
                if (a>0) if (history[a-1]==-1) break;
                hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(cla+1);

                for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
                hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
    	    }

    	    for (c=0; c<class_cn[cla]; c++) {
        	a=class_words[cla][c];

        	for (b=0; b<direct_order; b++) if (hash[b]) {
    		    neu2[a].ac+=syn_d[hash[b]];
            	    hash[b]++;
        	    hash[b]=hash[b]%direct_size;
    	        } else break;
    	    }
	}
        
        //activation 2   --softmax on words
	sum=0;
    	for (c=0; c<class_cn[cla]; c++) {
    	    a=class_words[cla][c];
    	    if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
    	    if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
    	    val=FAST_EXP(neu2[a].ac);
    	    sum+=val;
    	    neu2[a].ac=val;
    	}
    	for (c=0; c<class_cn[cla]; c++) neu2[class_words[cla][c]].ac/=sum;
	//
	
	f=random(0, 1);
        g=0;
        /*i=0;
        while ((g<f) && (i<vocab_size)) {
    	    g+=neu2[i].ac;
    	    i++;
        }*/
        for (c=0; c<class_cn[cla]; c++) {
    	    a=class_words[cla][c];
    	    g+=neu2[a].ac;
    	    if (g>f) break;
        }
        word=a;
        
	if (word>vocab_size-1) word=vocab_size-1;
        if (word<0) word=0;

	//printf("%s %d %d\n", vocab[word].word, cla, word);
	if (word!=0)
	    printf("%s ", vocab[word].word);
	else
	    printf("\n");

        copyHiddenLayerToInput();

        if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

        last_word=word;
	
	for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
	history[0]=last_word;
	
	if (independent && (word==0)) netReset();
        
        wordcn++;
    }
}
