///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

//这里的作用是防止rnnlmlib.h重复被include  
//如果程序第一次包含rnnlmlib.h,将会把#ifndef到文件最后一行的#endif之间的内容都执行  
//如果程序不是第一次包含rnnlmlib.h,则该文件的内容会被跳过  
#ifndef _RNNLMLIB_H_
#define _RNNLMLIB_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "hierarchical_cluster_discretizer.h"
//#include "hierarchical_cluster_fsthistory.h"

//real用于rnn中神经元的激活值,误差值类型  
typedef double real;            // doubles for NN weights
//direct_t表示最大熵模型中输入层到输出层权值类型  
typedef double direct_t;	// doubles for ME weights; TODO: check why floats are not enough for RNNME (convergence problems)

//最大字符串的长度 
#define MAX_STRING 200

typedef double real;
class FstHistory;
class Discretizer;

//rnn中神经元结构,两部分  
//ac表示激活值,er表示误差值,er用在网络学习时 
struct neuron 
{
    real ac;		//actual value stored in neuron
    real er;		//error value in neuron, used by learning algorithm
};

//突触,这里是表示网络层与层之间参数权值的结构  
//其实就是浮点类型,只是包上了一层,这样更形象                
struct synapse 
{
    real weight;	//weight of synapse
};

//这是一个word的结构定义
struct vocab_word 
{
    int cn;  //cn表示这个word在train_file中出现的频数
    char word[MAX_STRING];  //这个表示word本身,是字符串,但长度不能超过200

    //这个应该是在概率分布时表示当前词在历史下的条件概率  
    //但是后面的代码中我没看到怎么使用这个定义,感觉可以忽略 
    real prob;

    //这个表示当前词所在哪个类别
    int class_index;
};

//PRIMES[]这个数组装都是质数,质数的用处是来做散列函数的  
//对散列函数了解不多,个人理解可以使散列函数更少的冲突吧  
const unsigned int PRIMES[]={108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
//PRIMES数组长度,这个用法可以积累一下,以后自己的程序也可以使用
const unsigned int PRIMES_SIZE=sizeof(PRIMES)/sizeof(PRIMES[0]);

//最大阶数,这个是用来限制最大熵模型的N元模型特征的,N不能无穷大,这里最大是20  
const int MAX_NGRAM_ORDER=20;

//文件存储类型,TEXT表示ASCII存储,对存储网络权值时,有点浪费空间  
//BINARY表示二进制方式存储,对网络权值进行存储时,能更省空间,但是不便于阅读 
enum FileTypeEnum {TEXT, BINARY, COMPRESSED};		//COMPRESSED not yet implemented

//这个类就是RNN的结构定义 
class CRnnLM
{
protected:
    //训练数据集的文件名
    char train_file[MAX_STRING];
    //验证数据集的文件名
    char valid_file[MAX_STRING];
    //测试数据集的文件名
    char test_file[MAX_STRING];
    //RNN训练好后的模型所存储的文件
    char rnnlm_file[MAX_STRING];
    //其它语言模型对测试数据的生成文件，比如用srilm
    char lmprob_file[MAX_STRING];

    //随机种子,不同的rand_seed,可以导致网络权值初始化为不同的随机数  
    int rand_seed;
    
    //debug_mode分为两个级别,debug_mode>0会输出一些基本信息  
    //debug_mode>1会输出更详细的信息  
    int debug_mode;
    
    //rnn toolkit的版本号  
    int version;
    //用来指示存储模型参数时用TEXT, 还是用BINARY  
    int filetype;
    
    //控制开关,use_lmprob为0时表示不使用  
    //为1时表示使用了其他语言模型,并会将RNN和其他语言模型插值 
    int use_lmprob;
    //上面所说的插值系数  
    real lambda;
    //防止误差过大增长,用gradient_cutoff进行限制  
    //gradient_cutoff的使用在矩阵相乘那个函数里面可以看到 
    real gradient_cutoff;
    
    //dynamic如果大于0表示在测试时,边测试边学习 
    real dynamic;
    
    //学习率
    real alpha;
    //训练初始的学习率
    real starting_alpha;
    //变量控制开关,为0表明不将alpha减半,具体见代码  
    int alpha_divide;
    //logp表示累计对数概率,即logp = log10w1 + log10w2 + log10w3...  
    //llogp是last logp,即上一个logp  
    double logp, llogp;
    //最小增长倍数  
    float min_improvement;
    //iter表示整个训练文件的训练次数
    int iter;
    //vocab_max_size表示vocab最大容量,但是在代码中这个是动态增加的
    int vocab_max_size;
    //表示vocab的实际容量  
    int vocab_size;
    //记录train_file有多少word  
    int train_words;
    //指示当前所训练的词在train_file是第几个  
    int train_cur_pos;
    int counter;
    
    
    /////// HACK ///////
    int disc_map_set;
    Discretizer *d;
    FstHistory *fsth;
    ///////////////////
    
    //one_iter==1的话,只会训练一遍 
    int one_iter;
    //表示每训练anti_k个word,会将网络信息保存到rnnlm_file 
    int anti_k;
    
    //L2正规化因子  
    //实际在用的时候,是用的beta*alpha  
    real beta;
    
    //指定单词所分类别 
    int class_size;
    //class_words[i-1][j-1]表示第i类别中的第j个词在vocab中的下标 
    int **class_words;
    //class_cn[i-1]表示第i个类别中有多少word  
    int *class_cn;
    //class_max_cn[i-1]表示第i类别最多有多少word  
    int *class_max_cn;
    //old_classes大于0时用一种分类词的算法,否则用另一种
    int old_classes;
    
    //vocab里面存放的是不会重复的word,类型为vocab_word 
    struct vocab_word *vocab;

    //选择排序,将vocab[1]到vocab[vocab_size-1]按照他们出现的频数从大到小排序 
    void sortVocab();
    //里面存放word在vocab中的下标,这些下标是通过哈希函数映射来的  
    int *vocab_hash;
    //vocab_hash的大小
    int vocab_hash_size;
    
    //输入层的大小  
    int layer0_size;
    //隐藏层的大小
    int layer1_size;
    //压缩层的大小
    int layerc_size;
    //输出层的大小
    int layer2_size;
    
    //表示输入层到输出层直接连接的权值数组的大小  
    long long direct_size;
    //最大熵模型所用特征的阶数
    int direct_order;
    //history从下标0开始存放的是wt, wt-1,wt-2...
    int history[MAX_NGRAM_ORDER];
    
    //bptt<=1的话,就是常规的bptt,即只从st展开到st-1 
    int bptt;
    //每训练bptt_block个单词时,才会使用BPTT(或设置indenpendt不等于0,在句子结束时也可以进行BPTT)  
    int bptt_block;
    //bptt_history从下标0开始存放的是wt,wt-1,wt-2...  
    int *bptt_history;
    //bptt_hidden从下标0开始存放的是st,st-1,st-2...  
    neuron *bptt_hidden;
    //隐层到输入层的权值,这个使用在BPTT时的 
    struct synapse *bptt_syn0;
    
    int gen;
    
    //independent非0,即表示要求每个句子独立训练  
    //如果independent==0,表面上一个句子对下一个句子的训练时算作历史信息的  
    //这控制还得看句子与句子之间的相关性如何了  
    int independent;
    
    struct neuron *neu0;		//neurons in input layer
    struct neuron *neu1;		//neurons in hidden layer
    struct neuron *neuc;        //neurons in hidden layer
    struct neuron *neu2;		//neurons in output layer

    struct synapse *syn0;		//weights between input and hidden layer
    struct synapse *syn1;		//weights between hidden and output layer (or hidden and compression if compression>0)
    struct synapse *sync;		//weights between hidden and compression layer
    direct_t *syn_d;			//direct parameters between input and output layer (similar to Maximum Entropy model parameters)
    
    //backup used in training:
    struct neuron *neu0b;
    struct neuron *neu1b;
    struct neuron *neucb;
    struct neuron *neu2b;

    struct synapse *syn0b;
    struct synapse *syn1b;
    struct synapse *syncb;
    direct_t *syn_db;
    
    //backup used in n-bset rescoring:
    struct neuron *neu1b2;
    
    
public:

    int alpha_set, train_file_set;

    CRnnLM()		//constructor initializes variables
    {
        //这里的初始值只要初始是为非0的可以留意一下  
        version=10;
        filetype=TEXT;
        
        use_lmprob=0;
        lambda=0.75;
        gradient_cutoff=15;
        dynamic=0;
        
        /////// HACK //////
        disc_map_set = 0;
        d = NULL;
        fsth = NULL;
        //////////////////
        
        train_file[0]=0;
        valid_file[0]=0;
        test_file[0]=0;
        rnnlm_file[0]=0;
        
        alpha_set=0;
        train_file_set=0;
        
        alpha=0.1;
        beta=0.0000001;
        //beta=0.00000;
        alpha_divide=0;
        logp=0;
        llogp=-100000000;
        iter=0;
        
        min_improvement=1.003;
        
        train_words=0;
        train_cur_pos=0;
        vocab_max_size=100;
        vocab_size=0;
        vocab=(struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
        
        layer1_size=30;
        
        direct_size=0;
        direct_order=0;
        
        bptt=0;
        bptt_block=10;
        bptt_history=NULL;
        bptt_hidden=NULL;
        bptt_syn0=NULL;
        
        gen=0;
        
        independent=0;
        
        neu0=NULL;
        neu1=NULL;
        neuc=NULL;
        neu2=NULL;
        
        syn0=NULL;
        syn1=NULL;
        sync=NULL;
        syn_d=NULL;
        syn_db=NULL;
        //backup
        neu0b=NULL;
        neu1b=NULL;
        neucb=NULL;
        neu2b=NULL;
        
        neu1b2=NULL;
        
        syn0b=NULL;
        syn1b=NULL;
        syncb=NULL;
        //
        
        rand_seed=1;
        
        class_size=100;
        old_classes=0;
        
        one_iter=0;
        
        debug_mode=1;
        srand(rand_seed);
        //word映射为哈希的值小于100000000
        vocab_hash_size=100000000;
        vocab_hash=(int *)calloc(vocab_hash_size, sizeof(int));
    }
    
    ~CRnnLM()		//destructor, deallocates memory
    {
        int i;
        
        if (neu0!=NULL) 
        {
            free(neu0);
            free(neu1);
            if (neuc!=NULL) free(neuc);
            free(neu2);
            
            free(syn0);
            free(syn1);
            if (sync!=NULL) free(sync);
            
            if (syn_d!=NULL) free(syn_d);
            if (syn_db!=NULL) free(syn_db);

            //
            free(neu0b);
            free(neu1b);
            if (neucb!=NULL) free(neucb);
            free(neu2b);
            
            free(neu1b2);
            
            free(syn0b);
            free(syn1b);
            if (syncb!=NULL) free(syncb);
            //
            
            
            for (i=0; i<class_size; i++) free(class_words[i]);
            free(class_max_cn);
            free(class_cn);
            free(class_words);
        
            free(vocab);
            free(vocab_hash);
            
            if (bptt_history!=NULL) free(bptt_history);
            if (bptt_hidden!=NULL) free(bptt_hidden);
            if (bptt_syn0!=NULL) free(bptt_syn0);
            
            //todo: free bptt variables too
        }
    }
    
    //返回值类型为real且范围在[min, max]的数  
    real random(real min, real max);

	///////// HACK ////////
    void setDiscretizer(Discretizer *dis);
	//////////////////////    
	
    void setTrainFile(char *str);
    void setValidFile(char *str);
    void setTestFile(char *str);
    void setRnnLMFile(char *str);
    void setLMProbFile(char *str) {strcpy(lmprob_file, str);}
    
    void setFileType(int newt) {filetype=newt;}
    
    void setClassSize(int newSize) {class_size=newSize;}
    void setOldClasses(int newVal) {old_classes=newVal;}
    void setLambda(real newLambda) {lambda=newLambda;}
    void setGradientCutoff(real newGradient) {gradient_cutoff=newGradient;}
    void setDynamic(real newD) {dynamic=newD;}
    void setGen(real newGen) {gen=newGen;}
    void setIndependent(int newVal) {independent=newVal;}
    
    void setLearningRate(real newAlpha) {alpha=newAlpha;}
    void setRegularization(real newBeta) {beta=newBeta;}
    void setMinImprovement(real newMinImprovement) {min_improvement=newMinImprovement;}
    void setHiddenLayerSize(int newsize) {layer1_size=newsize;}
    void setCompressionLayerSize(int newsize) {layerc_size=newsize;}
    void setDirectSize(long long newsize) {direct_size=newsize;}
    void setDirectOrder(int newsize) {direct_order=newsize;}
    void setBPTT(int newval) {bptt=newval;}
    void setBPTTBlock(int newval) {bptt_block=newval;}
    void setRandSeed(int newSeed) {rand_seed=newSeed; srand(rand_seed);}
    void setDebugMode(int newDebug) {debug_mode=newDebug;}
    void setAntiKasparek(int newAnti) {anti_k=newAnti;}
    void setOneIter(int newOneIter) {one_iter=newOneIter;}
    
    struct neuron *getInputLayer() const { return neu0; }
    struct neuron *getHiddenLayer() const { return neu1; }
    struct neuron *getCompressionLayer() const { return neuc; }
    struct neuron *getOutputLayer() const { return neu2; }
    int getInputLayerSize() const { return layer0_size; }
    int getHiddenLayerSize() const { return layer1_size; }
    int getCompressionLayerSize() const { return layerc_size; }
    int getOutputLayerSize() const { return layer2_size; }
    int getClassSize() const { return class_size; }
    int getVocabSize() const { return vocab_size; }
    int getWordClass(int word) const { return vocab[word].class_index; }
    int getNumWordsInClass(int cl) const { return class_cn[cl]; }
    int getWordFromClass(int nth_w, int cl) const { return class_words[cl][nth_w]; }
    const char* getWordString(int word) const { return vocab[word].word; }
    int getWordCount(int word) const { return vocab[word].cn; }
    real getInputHiddenSynapse(int input_i, int hidden_i) { return syn0[input_i*layer1_size+hidden_i].weight; }
    
    //返回单词的哈希值  
    int getWordHash(char *word);
    //从文件中读取一个单词到word
    void readWord(char *word, FILE *fin);
    //查找word，找到返回word在vocab中的索引,没找到返回-1
    int searchVocab(char *word);
    //读取当前文件指针所指的单词,并返回该单词在vocab中的索引    
    int readWordIndex(FILE *fin);
    //将word添加到vocab中，并且返回刚添加word在vocab中的索引   
    int addWordToVocab(char *word);
    //从train_file中读数据,相关数据会装入vocab,vocab_hash    
    //这里假设vocab是空的 
    void learnVocabFromTrainFile();		//train_file will be used to construct vocabulary
    //保存当前的权值,以及神经元信息值  
    void saveWeights();			//saves current weights and unit activations
    //上面是暂存当前权值及神经元值，这里是从前面存下的数据中恢复    
    void restoreWeights();		//restores current weights and unit activations from backup copy
    //void saveWeights2();		//allows 2. copy to be stored, useful for dynamic rescoring of nbest lists
    //void restoreWeights2();
    //保存隐层神经元的ac值 		
    void saveContext();
    //恢复隐层神经元的ac值  
    void restoreContext();
    //保存隐层神经元的ac值    
    void saveContext2();
    //恢复隐层神经元的ac值  
    void restoreContext2();
    //初始化网络  
    void initNet();
    void saveNet();
    //从文件流中读取一个字符使其ascii等于delim    
    //随后文件指针指向delim的下一个 
    void goToDelimiter(int delim, FILE *fi);
    void restoreNet();
    //清除神经元的ac,er值 
    void netFlush();

    //隐层神经元(论文中的状态层s(t))的ac值置1    
    //s(t-1),即输入层layer1_size那部分的ac值置1    
    //bptt+history清0    
    void netReset();    //will erase just hidden layer state + bptt history + maxent history (called at end of sentences in the independent mode)
    
    //网络前向,计算概率分布
    void computeNet(int last_word, int word);
    void computeClassWordProbs(int last_word, int word);
    void computeClassProbs(int last_word);
    //反传误差,更新网络权值
    void learnNet(int last_word, int word);
    //将隐层神经元的ac值复制到输入层后layer1_size那部分
    void copyHiddenLayerToInput();
    void trainNet();
    void useLMProb(int use) {use_lmprob=use;}
    void testNet();
    void testNbest();
    void testGen();
    
    //矩阵和向量相乘  
    //1.type == 0时,计算的是神经元ac值,相当于计算srcmatrix × srcvec, 其中srcmatrix是(to-from)×(to2-from2)的矩阵 
    //srcvec是(to2-from2)×1的列向量,得到的结果是(to-from)×1的列向量,该列向量的值存入dest中的ac值  
    //2.type == 1, 计算神经元的er值,即(srcmatrix)^T × srcvec,T表示转置,转置后是(to2-from2)×(to-from),srcvec是(to-from)×1的列向量    
    void matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type);
};

#endif
