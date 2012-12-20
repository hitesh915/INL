function [ sample, sampleLabels ] = sus( population,labels,probabilities )
%SUS This function realize a sampling of the input data using the given
%certain probabilities of being selected.
%   INPUT:
%       - population: the all set of members of the population
%       - labels: the labels of each member of the population
%       - prbabilities: the specific probability of a member of the
%       population of being selected in the sample.
%   OUTPUT:
%       - sample: sample of the initial population
%       - sampleLabels: the label of each member of the sample.

    %Initialize the counters
    current = 1;
    i = 1;
    
    %The step of the selection sampling
    r = (1/size(population,1)).*rand(1,1);
    
    %An array with a comulative probabilitie
    prob = cumsum(probabilities);
    
    %Initializate the sample array
    sample = zeros(size(population,1),size(population,2));
    
    %Initializate the sample labels array
    sampleLabels = zeros(size(population,1),1);
    
    %Fill the sample array
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

