function [ kGroups ] = crossValPartition(groups, k, seed )
%crossValPartition Give k partitions of the given data
%   INPUT data:
%       - groups: groups to maintain in each partition
%       - k: number of folds
%       - seed: random seed
%   OUTPUT data:
%       - kGroups: k partitions of the data

    %Setup the random seed
    rng(seed);

    %Number of data
    numberOfPersons = size(unique(groups),1);
    
    %Number of persons per fold
    indPerFold = int32(floor(numberOfPersons/k));
    
    %Extra data to be folded
    modK = mod(numberOfPersons,k);
    
    %Partitions size indPerFold
    partition1 = zeros(indPerFold*(k-modK),1);
    
    %Partitions size indPerFold + 1
    partition2 = zeros((indPerFold+1)*modK,1);
        
    for i = 1:k-modK
        partition1(i*indPerFold-indPerFold+1:i*indPerFold,1) = i;
    end
    
    for i = 1:modK
        partition2(i*(indPerFold+1)-indPerFold:i*(indPerFold+1),1) = (k-modK)+i;
    end
        
    %Union the partition sets
    partition = [partition1; partition2];
        
    %Randomize partition set
    partition = partition(randperm(size(partition,1)));
        
    %Map each group to one partition
    map = containers.Map(unique(groups),partition);
    
    kGroups = arrayfun(@(x)values(map, {x}), groups, 'UniformOutput', true);
    kGroups = cell2mat(kGroups);
    
end