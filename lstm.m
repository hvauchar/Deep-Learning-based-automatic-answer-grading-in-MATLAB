clc;clear all;close all;
train = [];
[~,A] = xlsread('dataset/training_set_rel3.xlsx','C2:C1785');
T = xlsread('dataset/training_set_rel3.xlsx','D2:D1785');

%%
clc
e =length(A)*0.7;
e2 = length(A)*0.85;
e3 = length(A);
%%
clc
textDataTrain = A(1:e);
textDataValidation = A(round(e):round(e2));
textDataTest = A(round(e2):round(e3));
YTrain = T(1:e);
YValidation = T(e:round(e2));
YTest = T(round(e2):round(e3));
%%
figure
wordcloud(textDataTrain);
title("Training Data")
%%
clc
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation)
%%
documentsTrain(1:5)
%%
enc = wordEncoding(documentsTrain);
disp('done!')
%%
documentLengths = doclength(documentsTrain);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")
%%
XTrain = doc2sequence(enc,documentsTrain,'Length',75);
XTrain(1:5)
%%
XValidation = doc2sequence(enc,documentsValidation,'Length',75)
%%
inputSize = 1;
embeddingDimension = 100;
numWords = enc.NumWords;
numHiddenUnits = 180;

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(50)
    fullyConnectedLayer(20)
    fullyConnectedLayer(5)
    fullyConnectedLayer(1)
    regressionLayer]
%%
options = trainingOptions('adam', ...
    'MaxEpochs',15, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);
disp('done!')
%%
% net = trainNetwork(XTrain,YTrain,layers,options);
%%
load net_part1_bilstm
textDataTest = lower(textDataTest);
documentsTest = tokenizedDocument(textDataTest);
documentsTest = erasePunctuation(documentsTest);
XTest = doc2sequence(enc,documentsTest,'Length',75);
XTest(1:5)
YPred = predict(net,XTest)
%%

[round(YPred(1:5)),YTest(1:5)]
net_part1_bilstm = net
% save net_part1_bilstm
%%

acc = (sum([round(YPred)] == [YTest])./numel([YTest]))*length([YTest])
acc = sum(abs([round(YPred)] - [YTest]) == 1)./numel([YTest])
acc = sum(abs([round(YPred)] - [YTest]) == 2)./numel([YTest])
RMSE = sqrt(mean((round(YPred) - YPred).^2))





%%

% temp = "Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening."
temp = "Dear @CAPS1 @CAPS2, I believe that using computers will benefit us in many ways like talking and becoming friends will others through websites like facebook and mysace. Using computers can help us find coordibates, locations, and able ourselfs to millions of information. Also computers will benefit us by helping with jobs as in planning a house plan and typing a @NUM1 page report for one of our jobs in less than writing it. Now lets go into the wonder world of technology. Using a computer will help us in life by talking or making friends on line. Many people have myspace, facebooks, aim, these all benefit us by having conversations with one another. Many people believe computers are bad but how can you make friends if you can never talk to them? I am very fortunate for having a computer that can help with not only school work but my social life and how I make friends. Computers help us with finding our locations, coordibates and millions of information online. If we didn't go on the internet a lot we wouldn't know how to go onto websites that @MONTH1 help us with locations and coordinates like @LOCATION1. Would you rather use a computer or be in @LOCATION3. When your supposed to be vacationing in @LOCATION2. Million of information is found on the internet. You can as almost every question and a computer will have it. Would you rather easily draw up a house plan on the computers or take @NUM1 hours doing one by hand with ugly erazer marks all over it, you are garrenteed that to find a job with a drawing like that. Also when appling for a job many workers must write very long papers like a @NUM3 word essay on why this job fits you the most, and many people I know don't like writing @NUM3 words non-stopp for hours when it could take them I hav an a computer. That is why computers we needed a lot now adays. I hope this essay has impacted your descion on computers because they are great machines to work with. The other day I showed my mom how to use a computer and she said it was the greatest invention sense sliced bread! Now go out and buy a computer to help you chat online with friends, find locations and millions of information on one click of the button and help your self with getting a job with neat, prepared, printed work that your boss will love."
temp = preprocessText(temp);
temp = doc2sequence(enc,temp,'Length',75)
predict(net,temp)


