%% main.m - WiFi Broadcast Receiver PHY Prototype (802.11n)
% Полная цепочка обработки: захват -> синхронизация -> OFDM демодуляция -> декодирование
%
% Использование:
%   run_main('capture')   - захват I/Q данных с USRP B200
%   run_main('process')   - обработка захваченных данных
%   run_main('simulate')  - тест на симулированных данных

clear; clc; close all;

%% ==================== КОНФИГУРАЦИЯ ====================
cfg.centerFreq    = 2437e6;   % Частота канала 1 (2.437 GHz)
cfg.sampleRate    = 20e6;     % 20 MHz bandwidth
cfg.numSamples    = 4096;     % Количество сэмплов для захвата
cfg.channel       = 1;        % Номер WiFi канала
cfg.modulation    = 'QPSK';   % BPSK, QPSK, 16QAM, 64QAM
cfg.bw            = 20e6;     % Полоса пропускания
cfg.cyclicPrefix  = 80;       % 80 samples = 0.8 us (long CP)
cfg.Nfft          = 64;       % Размер FFT
cfg.numPackets    = 10;       % Количество пакетов для обработки

%% ==================== ВЫБОР РЕЖИМА ====================
mode = 'simulate';  % 'capture' | 'process' | 'simulate'

switch mode
    case 'simulate'
        % --- СИМУЛЯЦИЯ: генерация тестового 802.11n пакета ---
        fprintf('=== СИМУЛЯЦИЯ: Генерация тестового пакета 802.11n ===\n');
        [iqData, knownPreamble] = generate_80211n_packet(cfg);
        save('captured_samples.dat', 'iqData', '-binary');
        fprintf('Симулированный пакет сохранён.\n');
        
        % Обработка
        fprintf('\n=== ОБРАБОТКА ===\n');
        process_packet(iqData, cfg);
        
    case 'capture'
        % --- ЗАХВАТ С USRP B200 ---
        fprintf('=== ЗАХВАТ С USRP B200 ===\n');
        fprintf('Частота: %.1f MHz, Скорость: %.1f Msps\n', ...
            cfg.centerFreq/1e6, cfg.sampleRate/1e6);
        fprintf('Нажмите Ctrl+C для остановки...\n');
        
        try
            % Проверка наличия USRP
            usrp = uhd_device('usrpb210', cfg.sampleRate, 1);
            usrp.CenterFrequency = cfg.centerFreq;
            usrp.Gain = 30;  % LNA gain
            
            iqData = read(usrp, cfg.numSamples);
            save('captured_samples.dat', 'iqData', '-binary');
            fprintf('Данные захвачены: %d сэмплов\n', length(iqData));
            
            % Обработка
            process_packet(iqData, cfg);
            
        catch ME
            fprintf('Ошибка USRP: %s\n', ME.message);
            fprintf('Используйте GNU Radio для захвата:\n');
            fprintf('  python gnuradio/capture_samples.py\n');
        end
        
    case 'process'
        % --- ОБРАБОТКА ЗАХВАЧЕННЫХ ДАННЫХ ---
        fprintf('=== ОБРАБОТКА ЗАХВАЧЕННЫХ ДАННЫХ ===\n');
        if ~exist('captured_samples.dat', 'file')
            error('Файл captured_samples.dat не найден. Сначала выполните захват.');
        end
        load('captured_samples.dat', 'iqData');
        fprintf('Загружено %d сэмплов\n', length(iqData));
        process_packet(iqData, cfg);
end

%% ==================== ВИЗУАЛИЗАЦИЯ ====================
figure('Name', 'WiFi PHY Analysis', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

% Спектр
subplot(2,2,1);
N = length(iqData);
f = (-N/2:N/2-1) * cfg.sampleRate / N;
Y = fftshift(fft(iqData));
plot(f/1e6, 20*log10(abs(Y) + eps));
xlabel('Частота (MHz)'); ylabel('Мощность (dB)');
title('Спектр захваченного сигнала');
grid on; xlim([cfg.centerFreq/1e6 - 15, cfg.centerFreq/1e6 + 15]);

% IQ-диаграмма
subplot(2,2,2);
plot(real(iqData(1:500)), imag(iqData(1:500)), '.', 'MarkerSize', 1);
axis equal; grid on;
xlabel('In-phase'); ylabel('Quadrature');
title('IQ-диаграмма (первые 500 сэмплов)');

% Амплитуда
subplot(2,2,3);
plot(abs(iqData));
xlabel('Семпл'); ylabel('Амплитуда');
title('Амплитуда сигнала');
grid on;

% Фазовый шум
subplot(2,2,4);
phase = unwrap(angle(iqData(1:1000)));
plot(phase);
xlabel('Семпл'); ylabel('Фаза (rad)');
title('Фаза сигнала (первые 1000 сэмплов)');
grid on;

sgtitle('WiFi 802.11n PHY Analysis - USRP B200');
