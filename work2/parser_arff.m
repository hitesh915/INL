function [ features, classes ] = parser_arff( path )
    %PARSER_ARFF Parses a file containing a matrix
    %   This parser takes a text file and process it, removing the comments and
    %   type declarations and parsing the remaining matrix.
    %   The format of the matrix is expected to be a row per line, separated by
    %   commas. The last column (classes) is returned in the independent vector
    %   'classes' converted to their numeric indices, and all of the
    %   undefined values are converted to NaNs.

    
    function [row, class] = parse_row( string )
        srow = strread(string, '%s', 'delimiter', ',')';
        row = strrep(srow(1:end-1), '?', 'NaN');
        row = cellfun(@str2num, row);
        class = srow(end);
    end

    function [cnames, cnum] = parse_class_attribute(line)
        cnames = [];
        cnum = 0;
        
        % Check it defines an attribute
        if ~strncmpi(line, '@ATTRIBUTE', 10), return, end
        
        % Get attribute value
        s = strread(line,'%s');
        s = s(end);
        
        % Check value represents a set
        if ~strncmpi(s, '{', 1), return, end
        
        % Parse set classes
        s = s{1};
        s = s(2:end-1);
        cnames = strread(s, '%s', 'delimiter', ',')';
        
        % Count number of classes
        cnum = size(cnames);
        cnum = cnum(2);
    end

    % open file
    fp = fopen(path, 'r');
    
    % initialize values
    features = [];
    tclasses = [];
    cnames = [];
    cnum = [];
    
    % process file
    while ~feof(fp)
        line = fgetl(fp);
        if length(line)>0 && ~strncmpi(line, '%', 1)
            if strncmpi(line, '@', 1)
                [tcnames tcnum] = parse_class_attribute(line);
                if tcnum > 0
                    cnames  = tcnames;
                    cnum = tcnum;
                end
            else
                [tfeatures, tclass] = parse_row(line);
                features = [features ; tfeatures];
                tclasses = [tclasses tclass];
            end
        end
    end
    
    % count number of individuals
    nelems = size(tclasses);
    nelems = nelems(2);
    
    % replace class names by class indexs
    classes = [];
    for i=1:nelems
        cls = tclasses(i);
        for j=1:cnum
            if cell2mat(cls) == cell2mat(cnames(j))
                classes = [classes j];
            end
        end
    end
    
    
    % close file
    fclose(fp);
end

