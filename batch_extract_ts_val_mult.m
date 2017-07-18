%batch extract_ts_val

file_list = {...
    %'Extracted_0059_2016-02-26-16-28-27.mat' ...
    %'Extracted_0059_2016-02-26-16-38-13.mat' ...
    %'Extracted_504_2016-01-11-13-56-44.mat' ...
    %'Extracted_504_2016-01-11-14-10-01.mat' ...
    %'Extracted_0059_2017-02-09-12-52-17.mat' ...
    %'Extracted_0059_2017-02-09-13-46-37.mat' ...
    
    %'Extracted_504_2017-02-14-12-09-21.mat' ...
    %'Extracted_504_2017-02-14-12-35-41.mat' ...
    %'Extracted_504_2017-02-14-13-01-34.mat' ...
    
    %'Extracted_0059_2017-02-08-11-43-22.mat' ...
    %'Extracted_0059_2017-02-08-12-09-22.mat' ...
    %'Extracted_0059_2017-02-09-12-52-17.mat' ...
    %'Extracted_0059_2017-02-09-13-46-37.mat' ...
    %'Extracted_504_2017-02-08-10-36-11.mat' ...
    %'Extracted_504_2017-02-08-11-02-03.mat' ...
    %'Extracted_504_2017-02-09-11-50-03.mat' ...
    %'Extracted_504_2017-02-09-12-15-57.mat' ...
    'Extracted_504_2017-02-14-12-09-21.mat' ...
    'Extracted_504_2017-02-14-12-35-41.mat' ...
    'Extracted_504_2017-02-14-13-01-34.mat' ...
    
    };

for i = 1: length(file_list)
   filename = file_list{i};
    
   string = sprintf('extracting file %s', filename);
   disp(string)
   
   mult_extract_as_single(filename)
   
end
    