function [iqData, preamble] = generate_80211n_packet(cfg)
%% generate_80211n_packet - Генерация тестового 802.11n пакета
%
% Структура пакета 802.11n (HT-MPDU):
%   [SIGNAL] [SERVICE] [DATA] [TAIL]
%
% SIGNAL:  BPSK, 1 Mbps (длина, модуляция, кодирование)
% SERVICE: 12 бит заполнения (для синхронизации)
% DATA:    OFDM-символы с данными
% TAIL:    6 бит обнуления для декодера Витерби

Nfft = cfg.Nfft;          % 64 поднесущие
cpLen = cfg.cyclicPrefix; % 80 сэмплов (0.8 us)
bw = cfg.bw;              % 20 MHz
fs = cfg.sampleRate;      % 20 Msps

% --- Поднесущие 802.11a/n (64 точки FFT) ---
% Индексы: -32 .. 31
% DC: 0 (пустой)
% Левые крайние: -32..-25 (пустые)
% Правые крайние: 25..31 (пустые)
% Pilot: -21, -7, 7, 21
% Data: остальные

% Определяем индексы поднесущих
subcarrierIdx = -(Nfft/2):(Nfft/2-1); % -32 .. 31

% Пустые поднесущие (guard bands)
guardLeft  = subcarrierIdx <= -25;
guardRight = subcarrierIdx >= 25;
dcNull     = subcarrierIdx == 0;
nullSubs   = guardLeft | guardRight | dcNull;

% Pilot subcarriers (802.11a)
pilotIdx = [-21, -7, 7, 21];
pilotSubs = ismember(subcarrierIdx, pilotIdx);

% Data subcarriers
dataSubs = ~nullSubs & ~pilotSubs;
numDataSubs = sum(dataSubs); % 52 поднесущие для данных

fprintf('Генерация 802.11n пакета:\n');
fprintf('  FFT size: %d\n', Nfft);
fprintf('  Data subcarriers: %d\n', numDataSubs);
fprintf('  Pilot subcarriers: %d\n', sum(pilotSubs));
fprintf('  Cyclic prefix: %d samples (%.2f us)\n', cpLen, cpLen/fs*1e6);

% ==================== SIGNAL поле (LTF1) ====================
% Long Training Field - пилотные символы для оценки канала
% LTF1: известные значения на data subcarriers
% LTF2: комплексно-сопряжённые (для уточнения)

% Генерируем пилотные символы для LTF
% Для 802.11n HT-LTF используются те же пилоты что 802.11a
ltf1_data = ones(1, numDataSubs); % Известные значения
ltf2_data = -ones(1, numDataSubs);

% ==================== Генерация OFDM символов ====================

% --- LTF1 символ ---
ltf1_freq = zeros(1, Nfft);
ltf1_freq(dataSubs) = ltf1_data;
ltf1_freq(pilotIdx) = [1+1j, -1+1j, 1-1j, -1-1j]; % Pilot tones
ltf1_time = ifft(ltf1_freq, Nfft);
ltf1_time = [ltf1_time(end-cpLen+1:end), ltf1_time]; % Add CP

% --- LTF2 символ ---
ltf2_freq = zeros(1, Nfft);
ltf2_freq(dataSubs) = ltf2_data;
ltf2_freq(pilotIdx) = [1+1j, -1+1j, 1-1j, -1-1j];
ltf2_time = ifft(ltf2_freq, Nfft);
ltf2_time = [ltf2_time(end-cpLen+1:end), ltf2_time];

% --- SIGNAL поле (BPSK, 1 Mbps) ---
% 1 OFDM символ, BPSK, rate=1/2
signal_bits = [1 0 0 0 0 0 0 0 0 0 0 0]; % Length = 9 bits + 7 tail = 16 bits -> 2 OFDM symbols minimum
% Для простоты: 1 символ BPSK с известными данными
signal_data = ones(1, numDataSubs); % BPSK = 1
signal_freq = zeros(1, Nfft);
signal_freq(dataSubs) = signal_data;
signal_freq(pilotIdx) = [1+1j, -1+1j, 1-1j, -1-1j];
signal_time = ifft(signal_freq, Nfft);
signal_time = [signal_time(end-cpLen+1:end), signal_time];

% --- SERVICE поле (12 bits of zeros) ---
% 1 OFDM символ с нулевыми данными
service_data = zeros(1, numDataSubs);
service_freq = zeros(1, Nfft);
service_freq(dataSubs) = service_data;
service_freq(pilotIdx) = [1+1j, -1+1j, 1-1j, -1-1j];
service_time = ifft(service_freq, Nfft);
service_time = [service_time(end-cpLen+1:end), service_time];

% --- DATA поле (несколько OFDM символов) ---
% Генерируем случайные данные для тестирования
numDataSymbols = 5;
data_symbols = [];

for s = 1:numDataSymbols
    % Генерируем случайные биты
    numBitsPerSymbol = numDataSubs; % 1 bit/subcarrier for BPSK
    
    % Для QPSK/16QAM/64QAM - меняем модуляцию
    switch cfg.modulation
        case 'BPSK'
            bits = randi([0 1], 1, numBitsPerSymbol);
            symbols = 2*bits - 1; % BPSK: +1 or -1
        case 'QPSK'
            bits = randi([0 1], 2, numBitsPerSymbol/2);
            symbols = (-1).^(bits(1,:)) + 1j*(-1).^(bits(2,:));
        case '16QAM'
            bits = randi([0 1], 4, numBitsPerSymbol/4);
            symbols = qammod(bits', 16, 'InputType', 'bit', 'MapOrder', 'gray')';
        case '64QAM'
            bits = randi([0 1], 6, numBitsPerSymbol/6);
            symbols = qammod(bits', 64, 'InputType', 'bit', 'MapOrder', 'gray')';
    end
    
    % Обрезаем/дополняем до numDataSubs
    if length(symbols) > numDataSubs
        symbols = symbols(1:numDataSubs);
    elseif length(symbols) < numDataSubs
        symbols = [symbols, zeros(1, numDataSubs - length(symbols))];
    end
    
    % Раскладываем по поднесущим
    data_freq = zeros(1, Nfft);
    data_freq(dataSubs) = symbols;
    data_freq(pilotIdx) = [1+1j, -1+1j, 1-1j, -1-1j]; % Pilot tones
    
    % IFFT
    data_time = ifft(data_freq, Nfft);
    data_time = [data_time(end-cpLen+1:end), data_time]; % Add CP
    
    data_symbols = [data_symbols, data_time];
end

% ==================== Собираем пакет ====================
% Порядок: LTF1 + LTF2 + SIGNAL + SERVICE + DATA
preamble = [ltf1_time, ltf2_time];
header = signal_time;
packet = [preamble, header, service_time, data_symbols];

% Нормализация амплитуды
packet = packet / max(abs(packet)) * 0.8;

% Добавляем заголовок для идентификации
iqData = packet;

% Сохраняем структуру преамбулы для синхронизации
preamble = struct(...
    'ltf1', ltf1_time, ...
    'ltf2', ltf2_time, ...
    'pilotIdx', pilotIdx, ...
    'dataSubs', dataSubs, ...
    'Nfft', Nfft, ...
    'cpLen', cpLen, ...
    'numDataSubs', numDataSubs);

fprintf('Пакет сгенерирован: %d сэмплов\n', length(iqData));
end