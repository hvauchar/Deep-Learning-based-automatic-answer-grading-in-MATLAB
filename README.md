# Deep-Learning-based-automatic-answer-grading-in-MATLAB
This project presents a system for descriptive answer checking and grading application based on natural language processing and deep learning. The system is based on Bidirectional LSTM and word embedding to learn features of text using word encoding and trained them via Truncated Backpropagation(TBP)

Text data is naturally sequential. A piece of text is a sequence of words, which might have dependencies between them. To learn and use long-term dependencies to classify sequence data, use an LSTM neural network. An LSTM network is a type of recurrent neural network (RNN) that can learn long-term dependencies between time steps of sequence data.

To input text to an LSTM network, first convert the text data into numeric sequences. You can achieve this using a word encoding which maps documents to sequences of numeric indices. For better results, also include a word embedding layer in the network. Word embeddings map words in a vocabulary to numeric vectors rather than scalar indices. These embeddings capture semantic details of the words, so that words with similar meanings have similar vectors. They also model relationships between words through vector arithmetic. For example, the relationship "Rome is to Italy as Paris is to France" is described by the equation Italy – Rome + Paris = France.

1. There are four steps in training and using the LSTM network in this example:

2. Import and preprocess the data.

3. Convert the words to numeric sequences using a word encoding.

4. Create and train an Bi-LSTM network with a word embedding layer.

5. Classify new text data using the trained Bi-LSTM network.

![workflow](https://user-images.githubusercontent.com/49407332/74762228-6ec87800-52a3-11ea-8a6b-940fb95f440f.png)

![wordCloud](https://user-images.githubusercontent.com/49407332/81259369-70f5d380-9055-11ea-9eca-84791a17addb.png)
![LengthHistogram](https://user-images.githubusercontent.com/49407332/81259373-73f0c400-9055-11ea-9594-12522731d2be.png)

![network](https://user-images.githubusercontent.com/49407332/81259357-6d624c80-9055-11ea-9791-0011b7b4be4c.PNG)
![training](https://user-images.githubusercontent.com/49407332/81259363-6f2c1000-9055-11ea-91ff-bc74c6deb551.png)
