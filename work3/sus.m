function [ sample, sampleLabels ] = sus( population,labels,probabilities )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    current = 1;
    i = 1;
    
    r = (1/size(population,1)).*rand(1,1);
    prob = cumsum(probabilities);
    
    sample = zeros(size(population,1),size(population,2));
    sampleLabels = zeros(size(population,1),1);
    while current <= size(population,1)
        while r <= prob(i)
            sample(current,:) = population(i,:);
            sampleLabels(current) = labels(i);
            r = r + 1/size(population,1);
            current = current +1;
        end
        i = i + 1;
    end


end

