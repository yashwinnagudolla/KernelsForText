% Path to the directory containing the program files
data_dir = '/MATLAB Drive/';

% List all text files in the directory
file_names = dir(fullfile(data_dir, '*.txt'));
disp(file_names)

% Initialize a cell array to store all opcodes
all_opcodes = {};
opcode_counts = zeros(numel(file_names), 1);


% Combine all opcodes from the given programs
for i = 1:numel(file_names)
    file_path = fullfile(data_dir, file_names(i).name);
    file_opcodes = importdata(file_path);
    all_opcodes = [all_opcodes; file_opcodes];
end

categories = {'arithmetic', 'jump', 'data', 'num'};
mappings = {'arith', 'jump', 'data', 'num'}; 

mapped_opcodes = map_opcodes(all_opcodes, categories, mappings);
disp(mapped_opcodes);

% Create the new dictionary D2 using unique mapped opcodes
D2 = unique(mapped_opcodes);

document_vectors = zeros(numel(file_names), numel(D2));

% Represent each document as a vector using D2 (Vector Space Model)
for i = 1:numel(file_names)
    file_path = fullfile(data_dir, file_names(i).name);
    file_opcodes = importdata(file_path);
    document_vector = zeros(1, numel(D2));
    for j = 1:numel(D2)
        document_vector(j) = sum(strcmp(D2{j}, file_opcodes));
    end
    document_vectors(i, :) = document_vector;
end

% Term-Document Matrix (TDM)
term_document_matrix = document_vectors;

term_term_matrix = document_vectors' * document_vectors;

% Dot product kernel
dot_product_kernel = document_vectors * document_vectors';
%disp(dot_product_kernel)

% Polynomial kernel of degree 2
polynomial_kernel = (1 + dot_product_kernel).^2;
%disp(polynomial_kernel)
squared_euclidean_distance = 1 - dot_product_kernel;

% Apply K-means clustering to document vectors for dot product kernel
num_clusters = 4; 
[idx_dp, centroids_dp] = kmeans(squared_euclidean_distance, num_clusters);

% Apply K-means clustering to document vectors for polynomial kernel
[idx_poly, centroids_poly] = kmeans(polynomial_kernel, num_clusters);

% Display clustering results
disp('Clustering results using dot product kernel:');
disp(idx_dp);
disp('Clustering results using polynomial kernel (degree 2):');
disp(idx_poly);


class_labels = zeros(numel(file_names), 1);
for i = 1:numel(file_names)
    file_name = file_names(i).name;
    delimiter_index = strfind(file_name, '-');
    class_labels(i) = str2double(file_name(1:delimiter_index-1));
end

num_classes = length(unique(class_labels));

% Check for presence of diagonal sub-matrices
if exist('class_labels', 'var') && ~isempty(class_labels)
  disp('** Analysis of Diagonal Sub-Matrices (assuming class labels in file names) **');

  % Extract sub-matrices for each class
  class_submatrices = cell(1, num_classes);
  for class_id = 1:num_classes
    class_indices = find(class_labels == class_id);
    class_submatrices{class_id} = dot_product_kernel(class_indices, class_indices);
  end

  % Analyze similarity within each sub-matrix
  for class_id = 1:num_classes
    class_submatrix = class_submatrices{class_id};

    % Statistical measures
    mean_similarity = mean(class_submatrix(:));
    std_dev_similarity = std(class_submatrix(:));

    disp(['  - Class ', num2str(class_id), ':']);
    disp(['      Mean similarity: ', num2str(mean_similarity)]);
    disp(['      Standard deviation of similarity: ', num2str(std_dev_similarity)]);
    
    figure;
    imagesc(class_submatrix);
    colorbar;
    title(['Heatmap of Class ', num2str(class_id), ' Sub-Matrix']);
    xlabel('Programs');
    ylabel('Programs');

    % - Cosine similarity
    cosine_similarity = diag(pdist2(document_vectors(class_indices,:), document_vectors(class_indices,:),'cosine'));
    jaccard_similarity = zeros(size(class_submatrix));
    for i = 1:size(class_submatrix, 1)
      for j = i+1:size(class_submatrix, 1)
        intersection = sum(class_submatrix(i,:) & class_submatrix(j,:));
        union = sum(class_submatrix(i,:) | class_submatrix(j,:));
        jaccard_similarity(i, j) = intersection / union;
        jaccard_similarity(j, i) = jaccard_similarity(i, j);  % Symmetric matrix
      end
    end

    disp('      Jaccard similarity matrix (upper triangle):');
    disp(jaccard_similarity(tril(ones(size(jaccard_similarity))) > 0));  % Display only upper triangle for symmetry
  end
else
  disp(' - Diagonal sub-matrices cannot be confirmed without class labels.');
end


% Function to map opcodes to new tokens based on category
function mapped_opcodes = map_opcodes(all_opcodes, categories, mappings)
    mapped_opcodes = cell(size(all_opcodes));
    for i = 1:numel(all_opcodes)
        opcode = all_opcodes{i};
        found_category = false;
        switch true
            case any(contains(opcode, {'add', 'sub', 'mul', 'div'}))
                mapped_opcodes{i} = mappings{1}; % 'arith' category
                found_category = true;
            case startsWith(opcode, 'j')
                mapped_opcodes{i} = mappings{2}; % 'jump' category
                found_category = true;
            case any(contains(opcode, {'mov', 'push'}))
                mapped_opcodes{i} = mappings{3}; % 'data' category
                found_category = true;
            case ~isnan(str2double(opcode))
                mapped_opcodes{i} = mappings{4}; % 'num' category
                found_category = true;
        end
        % Use original opcode if no category match found
        if ~found_category
            mapped_opcodes{i} = opcode;
        end
    end
end

