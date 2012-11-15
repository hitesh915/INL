function [ matrix, classes ] = parser_arff( path )
    %PARSER_ARFF Parses a file containing a matrix
    %   This parser takes a text file and process it, removing the comments and
    %   type declarations and parsing the remaining matrix.
    %   The format of the matrix is expected to be a row per line, separated by
    %   commas. The last column (classes) is returned in the independent vector
    %   'classes' converted to their numeric indices, and all of the
    %   undefined values are converted to NaNs.

    cnames = [];
    cnum = 0;
    
    function [row, class] = parse_row( string )
        srow = strread(string, '%s', 'delimiter', ',')';
        row = strrep(srow(1:end-1), '?', 'NaN');
        row = cellfun(@str2num, row);
        
        % Read class
        class = srow(end);

        for i=1:cnum
            if strcmp(cell2mat(class), cell2mat(cnames(i)))
                row = [row i];
            end
        end
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
    matrix = [];
    
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
            elseif ~strncmpi(line, '@', 1)
                trow = parse_row(line);
                matrix = [matrix ; trow];
            end
        end
    end
    
    % Set list of classes return value
    classes = cnames;
    
    % close file
    fclose(fp);
end
