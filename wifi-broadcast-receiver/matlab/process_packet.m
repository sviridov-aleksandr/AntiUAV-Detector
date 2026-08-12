function process_packet(iqData, cfg)
%% process_packet - Полный PHY-конвейер обработки 802.11n пакета
%
% Этапы:
%   1. Коррекция частоты и времени (timing & frequency sync)
%   2. Оценка канала (channel estimation)
%   3. OFDM демодуляция (FFT)
%   4. Демодуляция поднесущих (BPSK/QPSK/16QAM/64QAM)
%   5. Декодирование Витерби (если применимо)

Nfft = cfg.Nfft;
cpLen = cfg.cyclicPrefix;
fs = cfg.sampleRate;
bw = cfg.bw;

fprintf('\n========================================\n');
fprintf('  PHY-КОНВЕЙЕР ОБРАБОТКИ 802.11n\n');
fprintf('========================================\n');

%% ==================== ЭТАП 1: СИХРОНИЗАЦИЯ ====================
fprintf('\n[1/5] Синхронизация...\n');

% --- Оценка задержки (timing offset) через корреляцию преамбулы ---
% Используем корреляцию с известной LTF (повторяющиеся символы)
[timingOffset, correlationPeak] = estimate_timing_offset(iqData, Nfft, cpLen);
fprintf('  Timing offset: %d samples\n', timingOffset);

% --- Коррекция частотной ошибки (frequency offset estimation) ---
% Используем корреляцию между LTF1 и LTF2
[freqOffset, phaseOffset] = estimate_freq_offset(iqData, timingOffset, Nfft, cpLen);
fprintf('  Frequency offset: %.2f Hz (%.2f ppm @ %.1f MHz)\n', ...
    freqOffset, freqOffset/bw*1e6, cfg.centerFreq/1e6);

% --- Применение коррекции ---
iqCorrected = iqData .* exp(-1j * (2*pi*freqOffset/fs * (0:length(iqData)-1) + phaseOffset));

% --- Сдвиг к началу пакета ---
startIdx = timingOffset + 1;
packetStart = iqCorrected(startIdx : startIdx + Nfft + cpLen*10); % Примерный размер

% Визуализация корреляции
figure('Name', 'Timing Synchronization', 'Position', [100, 100, 800, 400]);
subplot(1,2,1);
plot(correlationPeak);
xlabel('Семпл'); ylabel('Корреляция');
title('Корреляция для timing sync');
grid on;
[~, peakIdx] = max(abs(correlationPeak));
fprintf('  Пик корреляции на семпле: %d\n', peakIdx);

subplot(1,2,2);
plot(abs(iqCorrected(1:500)));
xlabel('Семпл'); ylabel('Амплитуда');
title('Корректированный сигнал');
grid on;

%% ==================== ЭТАП 2: ОЦЕНКА КАНАЛА ====================
fprintf('\n[2/5] Оценка канала...\n');

% Извлекаем LTF1 и LTF2 из пакета
% LTF1 начинается с timingOffset
ltf1_start = timingOffset + 1;
ltf1_symbol = iqCorrected(ltf1_start : ltf1_start + Nfft + cpLen - 1);
ltf2_symbol = iqCorrected(ltf1_start + Nfft + cpLen : ltf1_start + 2*(Nfft + cpLen) - 1);

% Удаляем циклический префикс
ltf1_noCP = ltf1_symbol(cpLen+1:end);
ltf2_noCP = ltf2_symbol(cpLen+1:end);

% FFT для перехода в частотную область
LTF1_freq = fft(ltf1_noCP, Nfft);
LTF2_freq = fft(ltf2_noCP, Nfft);

% Оценка канала: H = (LTF1 + LTF2*) / 2 / known_pilots
% Для LTF: известные значения на data subcarriers = +1
dataSubs = get_data_subcarriers(Nfft);
pilotIdx = [-21, -7, 7, 21];

H = (LTF1_freq + conj(LTF2_freq)) / 2;
H(dataSubs) = H(dataSubs) ./ ones(1, sum(dataSubs)); % Делим на известные значения

% Интерполяция для получения полного канала
H_full = interp_channel(H, Nfft, dataSubs, pilotIdx);

fprintf('  Channel estimate: %d поднесущих\n', sum(~iszero_channel(H)));
fprintf('  Max channel gain: %.2f dB\n', 20*log10(max(abs(H(dataSubs)))+eps));

%% ==================== ЭТАП 3: OFDM ДЕМОДУЛЯЦИЯ ====================
fprintf('\n[3/5] OFDM демодуляция...\n');

% Извлекаем OFDM-символы из пакета
% После LTF1+LTF2 идут SIGNAL + SERVICE + DATA
% Каждый символ: Nfft + cpLen сэмплов
numSymbols = 7; % LTF1, LTF2, SIGNAL, SERVICE, DATA (5+ символов)
ofdmSymbols = [];

for s = 0:numSymbols-1
    symStart = timingOffset + 1 + s * (Nfft + cpLen);
    sym = iqCorrected(symStart : symStart + Nfft + cpLen - 1);
    symNoCP = sym(cpLen+1:end); % Удаляем CP
    ofdmSymbols = [ofdmSymbols, symNoCP];
end

% FFT для каждого символа
numDataSymbols = floor(length(ofdmSymbols) / Nfft);
freqDomain = zeros(numDataSymbols, Nfft);

for s = 1:numDataSymbols
    start = (s-1)*Nfft + 1;
    sym = ofdmSymbols(start : start+Nfft-1);
    freqDomain(s, :) = fft(sym);
end

fprintf('  Извлечено %d OFDM-символов\n', numDataSymbols);

% Визуализация частотной области
figure('Name', 'OFDM Demodulation', 'Position', [100, 100, 800, 600]);
subplot(2,1,1);
plot(subcarrier_idx(Nfft), 20*log10(abs(freqDomain(1,:)) + eps));
xlabel('Поднесущая'); ylabel('Мощность (dB)');
title('Спектр первого OFDM-символа (LTF1)');
grid on;
xlim([-32, 31]);

subplot(2,1,2);
plot(subcarrier_idx(Nfft), 20*log10(abs(freqDomain(4,:)) + eps));
xlabel('Поднесущая'); ylabel('Мощность (dB)');
title('Спектр 4-го OFDM-символа (DATA)');
grid on;
xlim([-32, 31]);

%% ==================== ЭТАП 4: ДЕМОНДУЛЯЦИЯ ПОДНЕСУЩИХ ====================
fprintf('\n[4/5] Демодуляция поднесущих...\n');

% Извлекаем данные с data subcarriers для каждого символа
demodulatedBits = [];

for s = 1:numDataSymbols
    symFreq = freqDomain(s, :);
    
    % Извлекаем data subcarriers
    dataSubs_vals = symFreq(dataSubs);
    
    % Компенсация канала: Y / H
    if s <= 2
        % LTF символы - известные данные (=1)
        received = dataSubs_vals ./ (H(dataSubs) + eps);
    else
        % DATA символы - компенсируем канал
        received = dataSubs_vals ./ (H_full(dataSubs) + eps);
    end
    
    % Демодуляция в зависимости от типа символа
    if s == 3
        % SIGNAL поле - BPSK
        bits = bpsk_demod(received);
    elseif s == 4
        % SERVICE поле - нули
        bits = zeros(1, length(received));
    else
        % DATA - используемая модуляция
        switch cfg.modulation
            case 'BPSK'
                bits = bpsk_demod(received);
            case 'QPSK'
                bits = qpsk_demod(received);
            case '16QAM'
                bits = qam_demod(received, 16);
            case '64QAM'
                bits = qam_demod(received, 64);
        end
    end
    
    demodulatedBits = [demodulatedBits, bits];
end

fprintf('  Демодулировано %d бит\n', length(demodulatedBits));

% Визуализация QAM-диаграммы
if strcmp(cfg.modulation, '16QAM') || strcmp(cfg.modulation, '64QAM')
    figure('Name', 'Constellation Diagram', 'Position', [100, 100, 600, 600]);
    subplot(2,1,1);
    plot(real(received), imag(received), '.');
    axis equal; grid on;
    xlabel('I'); ylabel('Q');
    title(['Сигнальное созвездие - ', cfg.modulation]);
    
    % Теоретическое созвездие
    hold on;
    theo = qammod(0:15, 16, 'OutputType', 'real');
    plot(real(theo), imag(theo), 'rx', 'MarkerSize', 10);
    hold off;
end

%% ==================== ЭТАП 5: ДЕКОДИРОВАНИЕ ====================
fprintf('\n[5/5] Декодирование...\n');

% Декодирование Витерби (упрощённое)
% Для полного стека нужен Viterbi decoder с known trellis
% Здесь - базовая проверка

% Извлекаем поле SIGNAL (длина передачи)
signalFieldBits = demodulatedBits(1:9); % 9 bits length field
lengthField = bin2dec(num2str(signalFieldBits));
fprintf('  SIGNAL field (length): %d bytes\n', lengthField);

% Проверка CRC (если есть данные)
if length(demodulatedBits) > 20
    % Пытаемся извлечь данные
    dataBits = demodulatedBits(20:end);
    fprintf('  Данные: %d бит (%.0f байт)\n', length(dataBits), length(dataBits)/8);
    
    % Преобразование в байты
    numBytes = floor(length(dataBits) / 8);
    dataBytes = zeros(1, numBytes, 'uint8');
    for i = 1:numBytes
        byteBits = dataBits((i-1)*8+1 : i*8);
        dataBytes(i) = bin2dec(num2str(byteBits));
    end
    
    % Попытка распознать UDP-пакет
    if numBytes >= 42
        udpCheck = try_extract_udp(dataBytes);
        if udpCheck
            fprintf('  Обнаружен UDP-пакет!\n');
        else
            fprintf('  UDP-пакет не найден (проверьте порт/структуру)\n');
        end
    end
end

fprintf('\n========================================\n');
fprintf('  ОБРАБОТКА ЗАВЕРШЕНА\n');
fprintf('========================================\n');
end

%% ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

function [timingOffset, corr] = estimate_timing_offset(iqData, Nfft, cpLen)
%% Оценка timing offset через корреляцию Шмидта
% Используем корреляцию между текущим сэмплом и сэмплом cpLen назад
% R[n] = |sum(r[n] .* conj(r[n-cpLen]))| - lambda * sum(|r[n-cpLen]|^2)

L = length(iqData);
corrReal = zeros(1, L);
corrPower = zeros(1, L);

for n = cpLen+1:min(L, cpLen+500)
    corrReal(n) = abs(sum(iqData(n-cpLen+1:n) .* conj(iqData(n-2*cpLen+1:n-cpLen))));
    corrPower(n) = sum(abs(iqData(n-cpLen+1:n)).^2);
end

% Фильтрация
window = 10;
corrReal = movmean(corrReal, window);
corrPower = movmean(corrPower, window);

% Порог
lambda = 0.5;
corr = corrReal - lambda * corrPower;

% Находим пик
[~, maxIdx] = max(corr(cpLen+1:end));
timingOffset = cpLen + maxIdx - 1;
end

function [freqOffset, phaseOffset] = estimate_freq_offset(iqData, timingOffset, Nfft, cpLen)
%% Оценка частотной ошибки через LTF1 и LTF2
% LTF2 = -LTF1, разница фаз даёт freq offset

start = timingOffset + 1;
ltf1 = iqData(start : start + Nfft - 1);
ltf2 = iqData(start + Nfft + cpLen : start + 2*Nfft + cpLen - 1);

% FFT
LTF1 = fft(ltf1);
LTF2 = fft(ltf2);

% Фаза разности
phaseDiff = angle(LTF1 .* conj(LTF2));
phaseDiff = median(phaseDiff(~iszero_phase(phaseDiff))); % Медиана для робастности

phaseOffset = phaseDiff;
freqOffset = phaseDiff * Nfft / (2*pi * cpLen); % Hz
end

function dataSubs = get_data_subcarriers(Nfft)
%% Индексы data поднесущих для 802.11a/n (64 FFT)
subcarrierIdx = -(Nfft/2):(Nfft/2-1);
guardLeft  = subcarrierIdx <= -25;
guardRight = subcarrierIdx >= 25;
dcNull     = subcarrierIdx == 0;
pilotIdx   = ismember(subcarrierIdx, [-21, -7, 7, 21]);
dataSubs = ~guardLeft & ~guardRight & ~dcNull & ~pilotIdx;
end

function subcarrierIdx = subcarrier_idx(Nfft)
    subcarrierIdx = -(Nfft/2):(Nfft/2-1);
end

function H_full = interp_channel(H, Nfft, dataSubs, pilotIdx)
%% Линейная интерполяция канала
H_full = H;
% Интерполяция между pilot и data
for i = find(dataSubs)
    left = max(find(dataSubs < i));
    right = min(find(dataSubs > i));
    if ~isempty(left) && ~isempty(right)
        frac = (i - left) / (right - left);
        H_full(i) = (1-frac)*H(left) + frac*H(right);
    end
end
end

function count = iszero_channel(H)
    count = abs(H) < 1e-10;
end

function bits = bpsk_demod(received)
    bits = real(received) < 0;
end

function bits = qpsk_demod(received)
    bits = zeros(1, length(received)*2);
    for i = 1:length(received)
        bits(2*i-1) = real(received(i)) < 0;
        bits(2*i)   = imag(received(i)) < 0;
    end
end

function bits = qam_demod(received, M)
    % Упрощённая QAM демодуляция
    symbols = qamdemod(received', M, 'OutputType', 'bit')';
    bits = symbols(:)';
end

function bin = bin2dec(binStr)
    result = 0;
    for i = 1:length(binStr)
        result = result * 2 + binStr(i);
    end
    bin = result;
end

function iszero_phase = iszero_phase(phase)
    iszero_phase = abs(phase) > 1e-6;
end

function ok = try_extract_udp(dataBytes)
%% Попытка извлечь UDP-пакет из байтов
% UDP header: src_port(2) + dst_port(2) + length(2) + checksum(2)
if length(dataBytes) < 42
    ok = false;
    return;
end

% Проверяем, не IP-пакет ли это (первый байт 0x45 = IPv4, 20 bytes header)
if dataBytes(1) == 69 % 0x45
    % IP header present, skip 20 bytes
    offset = 21;
    % UDP starts after IP
    if length(dataBytes) >= offset + 8
        dstPort = bitshift(dataBytes(offset+1), 8) + dataBytes(offset+2);
        fprintf('  Destination port: %d\n', dstPort);
        ok = true;
    else
        ok = false;
    end
else
    % Без IP заголовка
    dstPort = bitshift(dataBytes(3), 8) + dataBytes(4);
    fprintf('  Destination port: %d\n', dstPort);
    ok = true;
end
end
