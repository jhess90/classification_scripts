function generate_save_unitwaveforms(chan, plxfile, num_of_examples, Freq) 

% eg: generate_save_unitwaveforms(Data_ts, plxfile, 100, 40000)
% Data_ts once formed can be sent in as input to generate 100 unit
% waveforms. 
% "Freq" is usually 40KHz. 

% for i = 1:length(Data_ts)
%     chan(i) = Data_ts(i).chan; 
% end
avg_wave = zeros(60,length(chan));

for i = 1:length(chan)
    [n, npw, ts, wave] = plx_waves_v(plxfile, chan(i), 1);
    total_wave(i).wave = wave;
    avg_wave(1:npw,i) = mean(wave); 
end

interest = 1:length(chan);

for i = 1:length(interest)
    wav_interest = []; 
    while length(wav_interest) < num_of_examples
        wav_interest = [wav_interest unique(randi(size(total_wave(interest(i)).wave,1),[1 100]))];
    end
    figure(i);
    plot(1/Freq:1/Freq:size(total_wave(interest(i)).wave,2)/Freq,total_wave(interest(i)).wave(wav_interest,:),'color',[0.5 0.5 0.5]);
    hold on;
    plot(1/Freq:1/Freq:size(total_wave(interest(i)).wave,2)/Freq,mean(total_wave(interest(i)).wave),'Linewidth',3,'color','k'); box off;
    title(['Unit ',num2str(interest(i))],'FontSize',24)
%     axis tight;
    box off;
%     set(gca,'YLim',[-0.12 0.12])
    saveas(gca,[plxfile(1:end-4),'_unit_',num2str(i),'.fig']);
    pause(1); 
    close all;
end