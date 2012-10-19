function [ matrix ] = parser_ariff( path )
    %PARSER Parses a file containing a matrix
    %   This parser takes a text file and process it, removing the comments and
    %   type declarations and parsing the remaining matrix.
    %   The format of the matrix is expected to be a row per line, separated by
    %   commas. The last column is discarded and all non-specified values
    %   are set to -1.
    
    function [trow] = parse_row( string )
        trow = strread(string, '%s', 'delimiter', ',')';
        trow = strrep(trow(1:end-1), '?', 'NaN');
        trow = cellfun(@str2num, trow);
    end

    % open file
    fp = fopen(path, 'r');
    
    % initialize values
    matrix = [];
    
    % process file
    while ~feof(fp)
        line = fgetl(fp);
        if length(line)>0 && ~strncmpi(line, '%', 1) && ~strncmpi(line, '@', 1)
            matrix = [matrix ; parse_row(line)];
        end
    end
    
    % close file
    fclose(fp);
end

