clear all
clc

% 基础路径
base_path = 'F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\' ;

% 力和频率参数
forces = {'0kN', '+10kN', '-10kN'}; % 力的情况
frequencies = [20, 40, 60];         % 频率情况

% 循环处理 G1 到 G8
for G = 6:8
    for sample = 1:9
        % 计算力和频率的索引
        force_index = ceil(sample / 3);
        freq_index = mod(sample-1, 3) + 1;

        % 构建文件路径
        sample_folder = sprintf('M0_G%d_LA0_RA0\\Sample_%d', G, sample);

        % 判断是否为 G3，若是，修改文件名
        if G == 3
            file_name = sprintf('data_gearbox_M0_G%d_LA0_RA0_%dHz_%s补.csv', G, frequencies(freq_index), forces{force_index});
        else
            file_name = sprintf('data_gearbox_M0_G%d_LA0_RA0_%dHz_%s.csv', G, frequencies(freq_index), forces{force_index});
        end

        full_path = fullfile(base_path, sample_folder, file_name);

        % 读取数据
        data = xlsread(full_path);
        data = data(:, 6); % 提取第6列数据

        % 动态生成保存路径和名称
        name = sprintf('M0_G%d_LA0_RA0_gearbox_CH18', G);
        save_path = fullfile('CWT3-1000', 'gearbox', 'test', sprintf('G%d', G), 'anomaly', sprintf('WC%d', sample));

        % 创建保存路径（如果不存在）
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end

        % 调用自定义函数
        Untitled2(data, save_path, name);
    end
end



%clear all
%clc
%
%% 基础路径
%base_path = 'F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets\M0_G0_LA0_RA0\';
%
%% 力和频率参数
%forces = {'0kN', '+10kN', '-10kN'}; % 力的情况
%frequencies = [20, 40, 60];         % 频率情况
%
%for sample = 1:9
%    % 计算力和频率的索引
%    force_index = ceil(sample / 3); % 每3个Sample对应一个力值
%    freq_index = mod(sample-1, 3) + 1; % 每3个Sample循环不同频率
%
%    % 文件名和完整路径
%    file_name = sprintf('Sample_%d\\data_gearbox_M0_G0_LA0_RA0_%dHz_%s.csv', ...
%                        sample, frequencies(freq_index), forces{force_index});
%    full_path = fullfile(base_path, file_name);
%
%    % 读取数据并提取第6列
%    data = xlsread(full_path);
%    data = data(:, 6);
%
%    % 动态保存路径和文件名
%    name = sprintf('M0_G0_LA0_RA0_gearbox_CH18_WC%d', sample);
%    save_path = 'CWT3-1000\gearbox\train\health';
%
%    % 调用原始函数
%    Untitled2(data, save_path, name);
%end

