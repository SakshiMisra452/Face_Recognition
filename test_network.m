loaded_Network = load('Face_Recognizer.mat');
net = loaded_Network.Trained_Network;

[Label, Probability] = classify(net, Resized_Validation_Data);
accuracy= mean(Label == Validation_Data.Labels);

index = randperm(numel(Validation_Data.Files), 4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(Validation_Data, index(i));
    imshow(I)
    label = Label(index(i));
    title(string(label) + ", " + num2str(100*max(Probability(index(i), :)), 3) + "%");
end