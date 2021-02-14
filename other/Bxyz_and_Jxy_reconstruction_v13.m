%%
clear all; close all;
%%% Load subfunctions
MyPath = userpath;
MyDir = MyPath(1:strfind(MyPath,';')-1);
MyWorkDir = genpath(strcat(MyDir,'func_Bxyz_recon'));
addpath(MyWorkDir, '-end');
MyWorkDir = genpath(strcat(MyDir,'Custom_scripts'));
addpath(MyWorkDir, '-end');

%% Parameters %%
%%%%%%%%%%%%%%%%
Path = 'C:\ExperimentalData\2021\2021-02\2021-02-12_D432_SiPV-20\ODMR - Pulsed_20_processed_bin_16';
% Path = 'C:\ExperimentalData\2020\2020-11\2020-11-11_SiPV-18_SpTmp\ODMR - Pulsed without IR_25_processed_bin_8'
Path_ref = '';
%Path = 'C:\ExperimentalData\2020\2020-10\2020-10-27_SiPV-18_newPS\ODMR - Pulsed without IR_56_processed_bin_8';

processor = 2;                      % =2 if processed with QDMPy, =1 if processed with python, =0 if processed with matlab
subtract_reference = 0;             % =1 to substract reference, otherwise only the signal image is analysed and plotted
subtract_B_subplane = 1;            % Subtract off the b plane from Bxyz for all further analysis, Sam added 2020/06/27
Rebinning = 1;                      % Additional binning when reconstructing Bxyz
freq_actually_measured = [1,1,1,1,1,1,1,1]; % set 1 (0) if the ODMR line was (was not) measured/fitted; the sum should match the # of freq maps available in the folder 

%%% Reconstruction
calc_J = 0;                         % =1 Do Jxyz calcs
reload_processed_data = 0;          % =1 Reload data rather than fitting again
reconstruction_method = 2;          % =0 direct inversion,  =1 fit Bxyz.uNV to BNVs,  =2 full ODMR fit using Hamiltonian
useXZ = 0;                          % in method #0 use either x (=1) or y (=0) BNVs to solve for z
BNV_used = [1,1,1,1];               % in method #1, set 1 the BNVs to use in the fit (choose at least 3)
freq_used = [1,1,1,0,0,1,1,1];      % in method #2, set 1 the freqs to use in the fit (at least as many as required for the fit function chosen)
Include_in_fit = 0;                 % in method #2, =0 just Zeeman, =1 hyperfine and quadrupole, 
                                    % =2 electric field (transverse components only since EZ cancel out with NV/VN), 
                                    % =3 approximation with E, =4 electric field taking the average of NV and VN
                                    % =5 electric field taking the average of NV and VN and including the N15 hyperfine
                                    % =6 hyperfine and electric field, =8 strain acting on D shift only, =9 strain full tensor
standoff = 1000e-9;                 % standoff distance assumed for J reconstruction
NV_layer_thickness = 100e-9;        % NV layer thickness used in J reconstruction
NV_above_or_below = 1;              % =1 if NV above source, =-1 otherwise
B_theta = -20;             % polar angle of the bias field (in deg)          
B_phi   = 150; % 166.15; % 170;            % azimuthal angle of the bias field (in deg)    -130
B_mag   = 80;

%%% fitting guesses
use_bounded_fit = 0;                % =1 use fminsearchbnd 
Bxyz_bounds = 100;                  % +/- for bounded fit (G)
D_guess = 2875;                     % Guess for the D parameter
D_guess_range = [2860,2890];        % Allowed range for D (MHz)
Exyz_guess = [0,0,0];               % Guess for the Exyz or shear strain
Exyz_LB = 1e5*[-0,-0,-10];          % Lower bound for E (V/cm)
Exyz_UB = 1e5*[0,0,10];             % upper bound for E (V/cm)

%%% Plotting
PlotODMRdata = 1;                   % =1 to plot ODMR data (freqs, contrast, FWHM, etc.)
auto_colour_range = 0;              % =1 for auto range, =0 to fix the range below
range_freq = 8;                   % freq maps will be capped to mean +/- range (in MHz)
range_D = 3;                      % D maps will be capped to mean +/- range (in MHz)
range_B = 40;                       % B maps will be capped to mean +/- range (in uT)
range_E = [0e5,1e5];                % E maps will be capped to this range
range_J = 100;                       % J maps will be capped to mean +/- range (in A/m)

%%% Linecuts
linecut_horizontal_or_vertical = 1; % =1 for horizontal, =2 for vertical
LinecutToPlot_horizontal = 4;      % Y index used to plot the example fit and the final linecut
LinecutToPlot_vertical = 4;        % X index used to plot the example fit and the final linecut
averaging_width = 1;                % number of +/- pixels to average over when plotting linecuts
Linecut_subplane = 0;               % =1 to plot linecuts from plane subtracted data, =0 otherwise
plot_linecuts_of_everything = 0;    % Plot linecut of all the other params

%%% Region of interest %%%
Full_ROI = 1;                       % =1 to fit the full image, =0 to use the specified square ROI specified below,
                                    % =2 to use a circular mask with centre and radius specified below
ROI_square = {6:38,1:64};     % if ROI is square, this is the area that will be fitted {Y,X}
ROI_centre = 1/1*[128,128];         % if ROI is a circular region, this is the centre of the circle (X,Y)
ROI_radius = 1/1*120;               % if ROI is a circular region, this is the radius of the circle

%%% Other options
plot_linecuts = 0;                  % =1 to plot linecuts
plot_individuals = 0;               % =1 to plot all the x,y,z projection in separate figures
plot_size_multiplier = 2;           % scaling factor in individual images
border_free = 0;                    % =1 to remove borders in individual images
plot_raw = 0;                       % =1 to plot raw B field in individual images (when no reference)
Bxyz_subtract_mean = 0;             % =1 subtracts the mean in plotting the raw Bxyz maps
rotate_xy = 0;                      % =1 to rotate the axes in the xy plane so that x->-y and y->x
plot_noncropped_data = 0;           % =1 to plot everything, 0 if plot only the FFT related stuff
plot_circular_mask = 0;             % =1 to plot Bz and |J| with circular mask, using centre and radius specified above
                                    % =2 to plot all B's and J's with circular mask

%%% Region used for plane substraction (or offset subtraction in FFT)
Full_subplane = 1;
x_subplane = [65:105];
y_subplane = [30:80];


%% Load data %%
%%%%%%%%%%%%%%%

init_Binning = str2double(Path(strfind(Path,'bin_')+4:end)); % Defines the binning from the file name for the pixle size
if init_Binning == 0
    init_Binning = 1;
end
Binning = init_Binning*Rebinning;

if Full_ROI >= 1
    ROI_square =  {0,0};
end

if processor == 1
    [Save_Path,PL,freq_mat,C_mat,df_mat,BNV_mat,BNV_mat_diff,I,current_on,...
    freq_mat_ref,C_mat_ref,df_mat_ref,BNV_mat_ref] = Bxyz_reconstruction_load_data_from_python(Path, Path_ref, reconstruction_method,Include_in_fit,Full_ROI,ROI_square,subtract_reference,Rebinning,freq_actually_measured);
elseif processor == 2
    [Save_Path,PL,freq_mat,C_mat,df_mat,BNV_mat,BNV_mat_diff,I,current_on,...
    freq_mat_ref,C_mat_ref,df_mat_ref,BNV_mat_ref] = Bxyz_reconstruction_load_data_from_QDMPy(Path, Path_ref, reconstruction_method,Include_in_fit,Full_ROI,ROI_square,subtract_reference,Rebinning,freq_actually_measured);
else
     [Save_Path,PL,freq_mat,C_mat,df_mat,BNV_mat,BNV_mat_diff,I,current_on,...
    freq_mat_ref,C_mat_ref,df_mat_ref,BNV_mat_ref] = Bxyz_reconstruction_load_data(Path, Path_ref, reconstruction_method,Include_in_fit,Full_ROI,ROI_square,subtract_reference,Rebinning);
end

if reload_processed_data == 1
    if subtract_reference == 1
        [Save_Path_sig,~,~,~,~,~,~,~,~,~,~,~,~] = Bxyz_reconstruction_load_data(Path, Path_ref, reconstruction_method,Include_in_fit,Full_ROI,ROI_square,0,Rebinning);
        [Save_Path_ref,~,~,~,~,~,~,~,~,~,~,~,~] = Bxyz_reconstruction_load_data(Path_ref, Path_ref, reconstruction_method,Include_in_fit,Full_ROI,ROI_square,0,Rebinning);
        [Bxyz_fit_sig, Exyz_fit_sig, D_fit_sig, error_mat_sig, PL_sig]  = Bxyz_reconstruction_reload_data(Save_Path_sig);
        [Bxyz_fit_ref, Exyz_fit_ref, D_fit_ref, error_mat_ref, PL_ref]  = Bxyz_reconstruction_reload_data(Save_Path_ref);
        for kk = 1:3
            Bxyz_fit_ref(:,:,kk) = imgaussfilt(Bxyz_fit_ref(:,:,kk),[20 20]);
        end
        Bxyz_fit = Bxyz_fit_sig-Bxyz_fit_ref;
        Exyz_fit = Exyz_fit_sig-Exyz_fit_ref;
        D_fit = D_fit_sig-D_fit_ref;
        error_mat = error_mat_sig-error_mat_ref;
        PL = PL_sig-PL_ref;
    else
        [Bxyz_fit, Exyz_fit, D_fit, error_mat, PL]  = Bxyz_reconstruction_reload_data(Save_Path);
    end
end
[nbY,nbX,nbL] =  size(freq_mat);

% Bxyz_fit(:,:,1) = Bxyz_fit(:,:,1)-3880e-2; 
% Bxyz_fit(:,:,2) = Bxyz_fit(:,:,2)-8540e-2; 
% Bxyz_fit(:,:,3) = Bxyz_fit(:,:,3)-2200e-2; 

%% Plot ODMR data %%
%%%%%%%%%%%%%%%%%%%%
colourmap = parula();
if Full_subplane == 1
    x_subplane = 1:nbX;
    y_subplane = 1:nbY;
end

if PlotODMRdata == 1
    plot_ODMR_data_2(Save_Path,freq_mat, C_mat,df_mat,BNV_mat,BNV_mat_diff,freq_mat_ref,BNV_mat_ref,...
                    subtract_reference,colourmap,auto_colour_range,range_B,range_D,range_freq,x_subplane,y_subplane);
end

savedata = BNV_mat(:,:,1);
save(strcat(Save_Path, '/BNV_1.txt'), 'savedata', '-ascii');

% return;

%% NV orientations and Bxyz guess %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[uNV, Bxyz_guess, freq_mat_for_fit, freq_mat_for_fit_ref, X, Y, Z] = NV_ori_and_Bxyz_guess(reconstruction_method,subtract_reference,...
    useXZ,BNV_mat,freq_mat,freq_mat_ref,B_theta,B_phi,B_mag);

%Bxyz_guess = [87.3,39.1,-20.9]; 

%% Process data

if reload_processed_data ~= 1
    
    %% Get (Bx,By,Bz) linecut %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [fitfunction, p0] = define_fit_function(reconstruction_method,...
        Include_in_fit, uNV, Bxyz_guess, D_guess, Exyz_guess);
    
 if reconstruction_method == 2
     if Include_in_fit == 0 || Include_in_fit == 1 
        LB = [Bxyz_guess-Bxyz_bounds, D_guess_range(1)];
        UB = [Bxyz_guess+Bxyz_bounds, D_guess_range(2)];
     elseif Include_in_fit == 9
        LB = [Bxyz_guess-Bxyz_bounds, D_guess_range(1), Exyz_LB, Exyz_LB];
        UB = [Bxyz_guess+Bxyz_bounds, D_guess_range(2), Exyz_UB, Exyz_UB];
     else
        LB = [Bxyz_guess-Bxyz_bounds, D_guess_range(1), Exyz_LB];
        UB = [Bxyz_guess+Bxyz_bounds, D_guess_range(2), Exyz_UB];
     end
 end
    
    %%% Modify the direction of the for loop based on vertical or horizontal linecuts
    if linecut_horizontal_or_vertical == 1
        ii = LinecutToPlot_horizontal;
        nb_linecut = nbX;
    else
        ii = LinecutToPlot_vertical;
        nb_linecut = nbY;
    end
    
    % Preallocation of memory
    Bxyz_fit_linecut = zeros(nb_linecut,4);
    Exyz_fit_linecut = zeros(nb_linecut,4);
    BNV_linecut_fit = zeros(nb_linecut,4);
    BNV_linecut_exp = zeros(nb_linecut,4);
    D_fit_linecut = zeros(nb_linecut);
    
    if reconstruction_method == 0 || reconstruction_method == 1
        mat_for_fit = BNV_mat;
    elseif reconstruction_method == 2
        mat_for_fit = freq_mat_for_fit;
    end
    
    if reconstruction_method == 0 %%% direct inversion method %%%
        for jj = 1:nb_linecut
            if linecut_horizontal_or_vertical == 1
                data_temp = reshape(mat_for_fit(ii,jj,:),[1,4]);
            else
                data_temp = reshape(mat_for_fit(jj,ii,:),[1,4]);
            end
            %%% Perform the dot product to extract out the B values.
            Bxyz_fit_linecut(jj,1) = dot(X,data_temp);
            Bxyz_fit_linecut(jj,2) = dot(Y,data_temp);
            Bxyz_fit_linecut(jj,3) = dot(Z,data_temp);
            BNV_linecut_exp(jj,:) = data_temp;
            D_fit_linecut(jj) = D_guess;
            
            %%% Do a standard fit just for the linecut only
            P = [Bxyz_fit_linecut(jj,1), Bxyz_fit_linecut(jj,2), Bxyz_fit_linecut(jj,3)];
            BNV_linecut_fit(jj,1:4) = fitfunction(P);
        end
    elseif reconstruction_method == 1 %%% fit method %%%
        for jj = 1:nb_linecut
            if linecut_horizontal_or_vertical == 1
                data_temp = reshape(mat_for_fit(ii,jj,:),[1,4]);
            else
                data_temp = reshape(mat_for_fit(jj,ii,:),[1,4]);
            end
            %%% perform fit and extract values
            err = @(p) sum((fitfunction(p).*BNV_used-data_temp.*BNV_used).^2);
%             p0 = Bxyz_guess;
            [P,err_val,exitflag] = fminsearch(err,p0);
            Bxyz_fit_linecut(jj,1:3) = P(1:3);
            BNV_linecut_fit(jj,:) = fitfunction(P);
            BNV_linecut_exp(jj,:) = data_temp;
            D_fit_linecut(jj) = D_guess;
        end
    elseif reconstruction_method == 2  %%% Full ODMR fit method %%%
        for jj = 1:nb_linecut
            if linecut_horizontal_or_vertical == 1
                data_temp = reshape(mat_for_fit(ii,jj,:),[1,8]);
            else
                data_temp = reshape(mat_for_fit(jj,ii,:),[1,8]);
            end
            %%% perform fit
            options = []; %optimset('MaxFunEvals',1e5);
            err = @(p) sum((fitfunction(p).*freq_used-data_temp.*freq_used).^2);
            if use_bounded_fit == 1
                [P,err_val,exitflag] = fminsearchbnd(err,p0,LB,UB,options);
            else 
                [P,err_val,exitflag] = fminsearch(err,p0);
            end
            %%% extract values
            Bxyz_fit_linecut(jj,1:3) = P(1:3);
            freq_fit = fitfunction(P);
            freq_linecut_fit(jj,1:8) = freq_fit;
            freq_linecut_prefit(jj,1:8) = fitfunction(p0);
            freq_linecut_exp(jj,1:8) = data_temp;
            D_fit_linecut(jj) = P(4);
            if Include_in_fit >= 2
                Exyz_fit_linecut(jj,1:3) = P(5:7);
                D_fit_linecut(jj) = P(4);
            end
        end
    end
    
    if LinecutToPlot_vertical == 2
        plot_vector = nbY;
    else
        plot_vector = nbX;
    end
    
    %%% Plot the BNV fits and values
    figure(6);
    color_vec = ['k','b','r','g','g','r','b','k'];
    if reconstruction_method == 2
        subplot(1,2,1); hold on;
        for ii = 1:8
            plot(1:plot_vector,freq_linecut_exp(:,ii),strcat(color_vec(ii),'o'),1:plot_vector,freq_linecut_prefit(:,ii),color_vec(ii));
        end
        xlabel('Pixel #'); ylabel('Frequency (MHz)'); title('Line cut of freq 1..8 - data vs intial guess');
        subplot(1,2,2); hold on;
        for ii = 1:8
            plot(1:plot_vector,freq_linecut_exp(:,ii),strcat(color_vec(ii),'o'),1:plot_vector,freq_linecut_fit(:,ii),color_vec(ii));
        end
        xlabel('Pixel #'); ylabel('Frequency (MHz)'); title('Line cut of freq 1..8 - data vs fit');
    else
        for ii = 1:4
            plot(1:plot_vector,BNV_linecut_exp(:,ii),strcat(color_vec(ii),'o'),1:plot_vector,BNV_linecut_fit(:,ii),color_vec(ii));
        end
        xlabel('Pixel #'); ylabel('B_{NV} (G)'); title('Line cut of BNV 1..4 - data vs fit');
    end
    set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,400,400]);
    hold off;
    saveas(figure(6),strcat(Save_Path,'/Linecuts.png'));
    
    %%% Plot the Bxyz linecuts
    figure(7);
    for ii = 1:3
        subplot(4,1,ii);
        plot(1:plot_vector,Bxyz_fit_linecut(:,ii),color_vec(ii));
    end
    subplot(4,1,4);
    plot(1:plot_vector,D_fit_linecut(1:end,1),color_vec(ii));
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,10,400,600]);
    subplot(4,1,1); title('Line cut of Bxyz and D - fit');
    
    %%% Plot the Exyz linecuts
    if reconstruction_method == 2 && Include_in_fit >= 2
        figure(8);
        for ii = 1:3
            subplot(3,1,ii);
            plot(1:plot_vector,Exyz_fit_linecut(:,ii),color_vec(ii));
        end
        set(gca,'FontSize',10); set(gcf,'units','points','position',[600,100,400,600]);
        subplot(3,1,1); title('Line cut of Exyz');
    end
    
    %% Get (Bx,By,Bz) map %%
    %%%%%%%%%%%%%%%%%%%%%%%%
%     p0 = P;
    %%% preallocate memory
    Bxyz_fit = zeros(nbY,nbX,3);
    Bxyz_fit_sig = zeros(nbY,nbX,3);
    Bxyz_fit_ref = zeros(nbY,nbX,3);
    Exyz_fit = zeros(nbY,nbX,3);
    error_map = zeros(nbY,nbX);
    D_fit = zeros(nbY,nbX);
    if reconstruction_method == 0
        progressbar()
        for ii = 1:nbY
            for jj = 1:nbX
                if subtract_reference == 1
                    data_temp = reshape(BNV_mat_diff(ii,jj,:),[1,4]);
                else
                    data_temp = reshape(BNV_mat(ii,jj,:),[1,4]);
                end
                Bxyz_fit(ii,jj,:) = [dot(X,data_temp), dot(Y,data_temp), dot(Z,data_temp)];
                error_mat(ii,jj) = 0;
            end
            Progression = ii/nbY;
            progressbar(Progression);
        end
    elseif reconstruction_method == 1
        parfor_progress(nbY)
        parfor ii = 1:nbY %#ok<*PFUIX>
            for jj = 1:nbX
                data_temp = reshape(BNV_mat(ii,jj,:),[1,4]);
                err = @(p) sum((fitfunction(p).*BNV_used-data_temp.*BNV_used).^2);
                [P,err_val,exitflag] = fminsearch(err,p0);
                Bxyz_fit(ii,jj,:) = P(1:3);
                D_fit(ii,jj) = D_guess;
                error_mat(ii,jj) = sqrt((1/4)*err_val);
                if subtract_reference == 1
                    data_temp_ref = reshape(BNV_mat_ref(ii,jj,:),[1,4]);
                    err = @(p) sum((fitfunction(p).*BNV_used-data_temp_ref.*BNV_used).^2);
                    [P_ref,err_val,exitflag] = fminsearch(err,p0);
                    error_mat_ref(ii,jj) = sqrt((1/4)*err_val);
                    data_temp_diff = reshape(BNV_mat(ii,jj,:)-BNV_mat_ref(ii,jj,:),[1,4]);
                    err = @(p) sum((fitfunction(p).*BNV_used-data_temp_diff.*BNV_used).^2);
                    [P_diff,err_val,exitflag] = fminsearch(err,p0);
                    error_mat(ii,jj) = sqrt((1/4)*err_val);
                    %%% extract values
                    Bxyz_fit_sig(ii,jj,:) = P(1:3);
                    Bxyz_fit_ref(ii,jj,:) = P_ref(1:3);
                    %Bxyz_fit(ii,jj,:) = P(1:3)-P_ref(1:3);
                    Bxyz_fit(ii,jj,:) = P_diff(1:3);
                end
            end
            %Progression = ii/nbY;
            %progressbar(Progression);
            parfor_progress;
        end
    elseif reconstruction_method == 2
        parfor_progress(nbY)
        parfor ii = 1:nbY
            for jj = 1:nbX
                data_temp = reshape(freq_mat(ii,jj,:),[1,8]);
                err = @(p) sum((fitfunction(p).*freq_used-data_temp.*freq_used).^2);
                if use_bounded_fit == 1
                    [P,err_val,exitflag] = fminsearchbnd(err,p0,LB,UB,options);
                else
                    [P,err_val,exitflag] = fminsearch(err,p0);
                end
                Bxyz_fit(ii,jj,:) = P(1:3);
                D_fit(ii,jj) = P(4);
                error_mat(ii,jj) = sqrt((1/8)*err_val);
                if Include_in_fit >= 2
                    Exyz_fit(ii,jj,:) = P(5:7);
                end
                if subtract_reference == 1
                    data_temp_ref = reshape(freq_mat_ref(ii,jj,:),[1,8]);
                    err = @(p) sum((fitfunction(p).*freq_used-data_temp_ref.*freq_used).^2);
                    if use_bounded_fit == 1
                        [P_ref,err_val,exitflag] = fminsearchbnd(err,p0,LB,UB,options);
                    else
                        [P_ref,err_val,exitflag] = fminsearch(err,p0);
                    end
                    D_fit_ref(ii,jj) = P_ref(4);
                    error_mat_ref(ii,jj) = sqrt((1/8)*err_val);
                    %%% extract values
                    Bxyz_fit_sig(ii,jj,:) = P(1:3);
                    Bxyz_fit_ref(ii,jj,:) = P_ref(1:3);
                    Bxyz_fit(ii,jj,:) = P(1:3)- P_ref(1:3);
                end
            end
            %Progression = ii/nbY;
            %progressbar(Progression);
            parfor_progress;
        end
    end
    
    if Bxyz_subtract_mean == 1
        Bxyz_fit(:,:,1) = Bxyz_fit(:,:,1) - nanmean(nanmean(Bxyz_fit(:,:,1)));
        Bxyz_fit(:,:,2) = Bxyz_fit(:,:,2) - nanmean(nanmean(Bxyz_fit(:,:,2)));
        Bxyz_fit(:,:,3) = Bxyz_fit(:,:,3) - nanmean(nanmean(Bxyz_fit(:,:,3)));
    end
    
end

%% Plot and save %%
%%%%%%%%%%%%%%%%%%%

if reload_processed_data == 1
% Bxyz_fit(:,:,1) = Bxyz_fit(:,:,1)-4330e-2; % 3890 
% Bxyz_fit(:,:,2) = Bxyz_fit(:,:,2)-9340e-2; % 8530
% Bxyz_fit(:,:,3) = Bxyz_fit(:,:,3)-2080e-2; % 2200
end
% Bxyz_fit(:,:,1) = Bxyz_fit(:,:,1)-0e-2; % 3890 
% Bxyz_fit(:,:,2) = Bxyz_fit(:,:,2)+20e-2; % 8530
% Bxyz_fit(:,:,3) = Bxyz_fit(:,:,3)-00e-2; % 2200
%subtract_reference = 1;

if rotate_xy == 1 
    temp = fliplr(Bxyz_fit(:,:,1)'); 
    Bxyz_fit(:,:,1) = -fliplr(Bxyz_fit(:,:,2)');
    Bxyz_fit(:,:,2) = -temp;
    Bxyz_fit(:,:,3) = fliplr(Bxyz_fit(:,:,3)');
    PL = fliplr(PL');
end

MatrixSize = size(squeeze(Bxyz_fit(:,:,1)));
RealSizePlot = MatrixSize*plot_size_multiplier;
title_vec = {'B_x (\muT)';'B_y (\muT)';'B_z (\muT)'};
title_vec_guass = {'B_x (G)';'B_y (G)';'B_z (G)'};
save_name_vec = ['Bx_mat';'By_mat';'Bz_mat'];
 
figure(10)
colour = redblue();
if subtract_reference == 1
    for ii = 1:3
        subplot(3,3,ii); B_map = Bxyz_fit_sig(:,:,ii); imagesc(B_map*1e2); 
        if auto_colour_range == 0
        	mid = mean(mean(B_map*1e2));
            caxis([mid-range_B,mid+range_B]);
        end  
        colorbar; axis equal; title(strcat(title_vec{ii},32,'- Signal')); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(3,3,ii+3); B_map = Bxyz_fit_ref(:,:,ii); imagesc(B_map*1e2); 
        if auto_colour_range == 0
        	mid = mean(mean(B_map*1e2));
            caxis([mid-range_B,mid+range_B]);
        end  
        colorbar; axis equal; title(strcat(title_vec{ii},32,'- Reference')); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(3,3,ii+6); B_map = Bxyz_fit(:,:,ii); imagesc(B_map*1e2); 
        if auto_colour_range == 0
        	mid = mean(mean(B_map*1e2));
            caxis([-range_B,range_B]);
        end  
        colorbar; axis equal; title(strcat(title_vec{ii},32,'- Difference')); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]); 
        save_name = strcat(Save_Path,'/',save_name_vec(ii),'.txt');
        save_data = Bxyz_fit(:,:,ii);
        save(save_name,'-ascii','save_data');
    end
    colormap(colour); set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,1000,800]);
else
    for ii = 1:3
        subplot(3,3,ii); imagesc(Bxyz_fit(:,:,ii)*1e2);
        if auto_colour_range == 0
        	mid = mean(mean(Bxyz_fit(:,:,ii)*1e2));
            caxis([mid-range_B,mid+range_B]);
        end  
        colorbar; axis equal; title(title_vec(ii)); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        save_name = strcat(Save_Path,'/',save_name_vec(ii),'.txt');
        save_data = Bxyz_fit(:,:,ii);
        save(save_name,'-ascii','save_data');
        [Bxyz_sub(:,:,ii), Bfit_bias(:,:,ii)] = subplane(Bxyz_fit(:,:,ii),x_subplane,y_subplane);
        subplot(3,3,ii+3); imagesc(Bxyz_sub(:,:,ii)*1e2);
        if auto_colour_range == 0
            caxis([-range_B,range_B]);
        end  
        colorbar; axis equal; title(strcat(title_vec{ii},32,'- subplane')); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(3,3,ii+6); imagesc(Bfit_bias(:,:,ii));
        colorbar; axis equal; title(strcat(title_vec_guass{ii},32,'- B fit')); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        save_name = strcat(Save_Path,'/',save_name_vec(ii,:),'_subplane.txt');
        save_data = Bxyz_sub(:,:,ii);
        save(save_name,'-ascii','save_data');
    end
    colormap(colour); set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,1000,600]);
end
saveas(figure(10),strcat(Save_Path,'\Bxyz.png'));

if reload_processed_data ~= 1
    datatosave = Bxyz_fit(:,:,1); save(strcat(Save_Path, '/Bx_mat.txt'), 'datatosave' , '-ascii');
    datatosave = Bxyz_fit(:,:,2); save(strcat(Save_Path, '/By_mat.txt'), 'datatosave' , '-ascii');
    datatosave = Bxyz_fit(:,:,3); save(strcat(Save_Path, '/Bz_mat.txt'), 'datatosave' , '-ascii');
end

if ~subtract_reference && subtract_B_subplane
    Bxyz_fit(:, :, 1) = Bxyz_sub(:, :, 1);
    Bxyz_fit(:, :, 2) = Bxyz_sub(:, :, 2);
    Bxyz_fit(:, :, 3) = Bxyz_sub(:, :, 3);
end

if reconstruction_method == 2
    figure(15)
    if subtract_reference == 1
        subplot(1,2,1); imagesc(D_fit);
        if auto_colour_range == 0
        	mid = mean(mean(D_fit));
            caxis([mid-range_D,mid+range_D]);
        end  
        colorbar; axis equal; title('D (MHz) from full ODMR fit - signal'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(1,2,2); imagesc(D_fit_ref);
        if auto_colour_range == 0
        	mid = mean(mean(D_fit_ref));
            caxis([mid-range_D,mid+range_D]);
        end  
        colorbar; axis equal; title('D (MHz) from full ODMR fit - reference'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,700,250]);
    else
        imagesc(D_fit);
        if auto_colour_range == 0
        	mid = mean(mean(D_fit));
            caxis([mid-range_D,mid+range_D]);
        end 
        colorbar; axis equal; title('D (MHz) from full ODMR fit'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        %colormap(colour);
        set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,320,250]);
    end
    saveas(figure(15),strcat(Save_Path,'/D_fit.png'));
    datatosave = D_fit; save(strcat(Save_Path, '/D_mat.txt'), 'datatosave' , '-ascii');
end

if reconstruction_method == 2 && (Include_in_fit >= 2)
    figure(20)
    for ii = 1:3
        subplot(1,3,ii)
        if auto_colour_range == 1
            imagesc(Exyz_fit(:,:,ii));
        else
            imagesc(Exyz_fit(:,:,ii),range_E);
        end
        colorbar; axis equal; colormap(gray);
        title('Exyz fits from full ODMR fit'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    end
    for ii = 1:3
%         subplot(2,3,3+ii)
%         %         Exyz_nanmean = nanmean(nanmean(Exyz_fit(:,:,ii)));
%         Exyz_sub(:,:,ii) = Exyz_fit(:,:,ii);
%         %         Exyz_sub(:,:,ii) = subplane(Exyz_fit(:,:,ii), [8e6 1.4e4 7e3]);
%         %         Exyz_sub(:,:,ii) = subparabola(Exyz_fit(:,:,ii), [8e6 1.4e4 7e3 -10 -10]);
%         %           Exyz_sub(:,:,ii) =  Exyz_sub(:,:,ii) -  smooth(smooth(nanmean(Exyz_sub([10:30 80:end],:,ii))))';
%         if auto_colour_range == 1
%             imagesc(Exyz_sub(:,:,ii));
%         else
%             imagesc(Exyz_sub(:,:,ii),range_E);
%         end
%         colorbar; axis equal; colormap(gray);
%         title('Exyz fits from full ODMR fit'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);   
    end
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,100,1200,400]);
	saveas(figure(20),strcat(Save_Path,'/Exyz_fit.png'));    
    
    datatosave = Exyz_fit(:,:,1); save(strcat(Save_Path, '/Ex_mat.txt'), 'datatosave' , '-ascii');
    datatosave = Exyz_fit(:,:,2); save(strcat(Save_Path, '/Ey_mat.txt'), 'datatosave' , '-ascii');
    datatosave = Exyz_fit(:,:,3); save(strcat(Save_Path, '/Ez_mat.txt'), 'datatosave' , '-ascii');
%     datatosave = Exyz_sub(:,:,1); save(strcat(Save_Path, '/Ex_mat_subplane.txt'), 'datatosave' , '-ascii');
%     datatosave = Exyz_sub(:,:,2); save(strcat(Save_Path, '/Ey_mat_subplane.txt'), 'datatosave' , '-ascii');
%     datatosave = Exyz_sub(:,:,3); save(strcat(Save_Path, '/Ez_mat_subplane.txt'), 'datatosave' , '-ascii');
end

if reconstruction_method >= 1
    figure(16)
    title_vec = {'fit error (G)';'fit error (MHz)';'fit error (G) - reference';'fit error (MHz) - reference'};
    if subtract_reference == 1
        subplot(1,2,1); imagesc(error_mat); 
        colorbar; axis equal; colormap(parula); title(title_vec{reconstruction_method}); 
        axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(1,2,2); imagesc(error_mat_ref);
        colorbar; axis equal; colormap(parula); title(title_vec{reconstruction_method+2}); 
        axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,700,250]);
    else
        imagesc(error_mat); 
        colorbar; axis equal; colormap(parula); title(title_vec{reconstruction_method});
        axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,320,250]);
    end
    saveas(figure(16),strcat(Save_Path,'/fit_error.png'));
    datatosave = error_mat; save(strcat(Save_Path, '/fit_error_mat.txt'), 'datatosave' , '-ascii');
end

if plot_noncropped_data  
    figure(17)
    imagesc(PL);
    colorbar; axis equal; colormap(gray);
    axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,100,600,400]);
    set(gca,'position',[0 0 1 1],'units','normalized'); axis off;
    set(gca,'LooseInset',get(gca,'TightInset'));
    set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
    if border_free == 1
        hFigure = figure(17);
        set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
    end
    saveas(figure(17),strcat(Save_Path,'/PL.png'));
end
datatosave = PL; save(strcat(Save_Path, '/PL.txt'), 'datatosave' , '-ascii');

if plot_noncropped_data == 1 && plot_individuals == 1

    if subtract_reference == 1
        for ii = 1:3
            figure(10+ii)
            if auto_colour_range == 1
                imagesc(Bxyz_fit(:,:,ii)*1e2);
            else
                imagesc(Bxyz_fit(:,:,ii)*1e2,[-range_B,range_B]);
            end
            colorbar; axis equal; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
            colormap(colour); set(gca,'FontSize',10); 
            set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
            if border_free == 1
                hFigure = figure(10+ii);
                set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
                set(gca,'LooseInset',get(gca,'TightInset'));
                set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
            end
        end

    else
        for ii = 1:3
            figure(10+ii)
            Bxyz_sub(:,:,ii) = subplane(Bxyz_fit(:,:,ii),x_subplane,y_subplane);
            if auto_colour_range == 1
                if plot_raw == 1
                    imagesc(Bxyz_fit(:,:,ii)*1e2); axis equal; colorbar; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
                else
                    imagesc(Bxyz_sub(:,:,ii)*1e2); axis equal; colorbar; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
                end
            else
                if plot_raw == 1
                    menanmean = nanmean(nanmean(Bxyz_fit(:,:,ii)*1e2));
                    imagesc(Bxyz_fit(:,:,ii)*1e2,[-range_B,range_B]); colorbar; title(title_vec(ii)); set(gca,'xtick',[]);
                else
                    imagesc(Bxyz_sub(:,:,ii)*1e2,[-range_B,range_B]); colorbar; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
                end
            end
            colormap(colour); axis equal; set(gca,'FontSize',10);
            set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
            if border_free == 1
                hFigure = figure(21+ii);
                set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
                set(gca,'LooseInset',get(gca,'TightInset'));
                set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
            end
        end
    end
    saveas(figure(11),strcat(Save_Path,'\Bx.png'));
    saveas(figure(12),strcat(Save_Path,'\By.png'));
    saveas(figure(13),strcat(Save_Path,'\Bz.png'));
    
    for ii = 1:3
        figure(20+ii)
        if auto_colour_range == 1
            imagesc(Exyz_fit(:,:,ii)); colorbar; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
        else
            imagesc(Exyz_fit(:,:,ii),range_E); colorbar; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
        end
        colormap(gray); axis equal; set(gca,'FontSize',10);
        set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
            if border_free == 1
                hFigure = figure(20+ii);
                set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
                set(gca,'LooseInset',get(gca,'TightInset'));
                set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
            end
    end
    saveas(figure(21),strcat(Save_Path,'\Ex.png'));
    saveas(figure(22),strcat(Save_Path,'\Ey.png'));
    saveas(figure(23),strcat(Save_Path,'\Ez.png'));
    
end

%% plot linecuts of other params
if plot_linecuts_of_everything == 1
    
    horx = LinecutToPlot_horizontal;
    verx = LinecutToPlot_vertical;
    
    %%% PL
    figure(26)
    
    [avg_hor, avg_ver, hor_Lim, ver_Lim] = take_linecut(horx ,verx, averaging_width,  PL) ;
    subplot(1,2,1)
    plot(avg_hor); title('horizontal PL linecut'); %axis([X_Lim Con_hor_Lim]);
    subplot(1,2,2)
    plot(avg_ver); title('vertical PL linecut'); %axis([X_Lim Con_ver_Lim]);
    saveas(figure(26),strcat(Save_Path,'\PL linecuts.png'));
    
    datatosave = avg_hor';
    save(strcat(Save_Path, '/PL_hor_linecuts.txt'), 'datatosave' , '-ascii');
    datatosave = avg_ver;
    save(strcat(Save_Path, '/PL_ver_linecuts.txt'), 'datatosave' , '-ascii');
    %%% Contrast
    figure(27)
    for ii = 1:8
        [avg_hor, avg_ver, hor_Lim, ver_Lim] = take_linecut(horx ,verx, averaging_width, C_mat(:,:,ii)) ;
        subplot(1,2,1)
        plot((x_vec-x_offset)*1e6,avg_hor); title('horizontal Contrast linecut'); %axis([X_Lim hor_Lim]);
        hold on
        subplot(1,2,2)
        plot((y_vec-y_offset)*1e6,avg_ver); title('vertical Contrast linecut'); %axis([X_Lim ver_Lim]);
        hold on
    end
    legend('C1','C2','C3','C4','C5','C6','C7','C8','location','best')
    hold off
    hold off
    saveas(figure(27),strcat(Save_Path,'\Contrast linecuts.png'));
    
    
    %%% ODMR Width
    figure(28)
    for ii = 1:8
        [avg_hor, avg_ver, hor_Lim, ver_Lim] = take_linecut(horx ,verx, averaging_width, df_mat(:,:,ii)) ;
        subplot(1,2,1)
        plot((x_vec-x_offset)*1e6,avg_hor); title('horizontal ODMR width linecut'); %axis([X_Lim hor_Lim]);
        hold on
        subplot(1,2,2)
        plot((y_vec-y_offset)*1e6,avg_ver); title('vertical ODMR width linecut'); %axis([X_Lim ver_Lim]);
        hold on
    end
    legend('df1','df2','df3','df4','df5','df6','df7','df8','location','best')
    hold off
    hold off
    saveas(figure(28),strcat(Save_Path,'\ODMR width linecuts.png'));
    
    
    %%% BNV
    figure(29)
    for ii = 1:4
        [avg_hor, avg_ver, hor_Lim, ver_Lim] = take_linecut(horx ,verx, averaging_width, BNV_mat(:,:,ii)) ;
        subplot(1,2,1)
        plot((x_vec-x_offset)*1e6,avg_hor-nanmean(avg_hor)); title('horizontal BNV linecut'); %axis([X_Lim hor_Lim]);
        hold on
        subplot(1,2,2)
        plot((y_vec-y_offset)*1e6,avg_ver-nanmean(avg_ver)); title('vertical BNV linecut'); %axis([X_Lim ver_Lim]);
        hold on
    end
    legend('BNV1','BNV2','BNV3','BNV4','location','best')
    hold off
    hold off
    saveas(figure(29),strcat(Save_Path,'\BNV linecuts.png'));
    
    
end

savedata = Bxyz_fit(:,:,1); save(strcat(Save_Path, '/full_Bx.txt'), 'savedata', '-ascii');
savedata = Bxyz_fit(:,:,2); save(strcat(Save_Path, '/full_By.txt'), 'savedata', '-ascii');
savedata = Bxyz_fit(:,:,3); save(strcat(Save_Path, '/full_Bz.txt'), 'savedata', '-ascii');
savedata = sqrt(sum(Bxyz_fit.^2,3)); save(strcat(Save_Path, '/full_Bmag.txt'), 'savedata', '-ascii');



%% Extrapolate data %%
%%%%%%%%%%%%%%%%%%%%%%

if subtract_reference ~= 1
        Bxyz_fit(:,:,1) = Bxyz_fit(:,:,1) - nanmean(nanmean(Bxyz_fit(:,:,1)));
        Bxyz_fit(:,:,2) = Bxyz_fit(:,:,2) - nanmean(nanmean(Bxyz_fit(:,:,2)));
        Bxyz_fit(:,:,3) = Bxyz_fit(:,:,3) - nanmean(nanmean(Bxyz_fit(:,:,3)));
end
    
% Bxyz_fit(:,:,1) = 4/5*Bxyz_fit(:,:,1)+0e-2; 
% Bxyz_fit(:,:,2) = 4/5*Bxyz_fit(:,:,2)-0e-2; 
% Bxyz_fit(:,:,3) = Bxyz_fit(:,:,3)+8e-2;

pixelsize_x = 2*Binning*108e-9; pixelsize_y = 2*Binning*108e-9;       % Argus with 300 mm lens: 108 nm = 6.5 um / 40x mag x 200 mm / 300 mm
                                                                    % Zyla with 500 mm lens: 65 nm = 6.5 um / 40x mag x 200 mm / 500 mm
                                                                    % atto with 300 mm lens: 62 nm = 6.5 um * 2.87 mm focal length / 300 mm 
SizeFactorForFFT_x = 1;
SizeFactorForFFT_y = 1;
padded = 0;
mu0 = 4e-7*pi; 
crop = 0;
avg_extra = 10;
avg_width = 10;

B_data(:,:,1:3) = Bxyz_fit(:,:,:);
B_data(:,:,4:7) = BNV_mat(:,:,:); 
B_data_extraX = zeros(nbY,SizeFactorForFFT_x*nbX,7);            
B_data_extra = zeros(SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX,7);
% for kk=1:7
%     Bxyz_fit_smooth(:,:,kk) = imgaussfilt(Bxyz_fit(:,:,kk),[20 20]);
%     Bxyz_fit_smooth(:,:,kk) = Bxyz_fit(:,:,kk);
% end

% Extrapolate in x
if SizeFactorForFFT_x == 1
    B_data_extraX(:,:,:) = B_data(:,:,:);
else    
    B_data_extraX(:,SizeFactorForFFT_x*nbX/2-nbX/2+1:SizeFactorForFFT_x*nbX/2+nbX/2,:) = B_data(:,:,:);
%    slope_right = nanmean(Bxyz_fit_smooth(:,end-avg_extra:end,:)-Bxyz_fit_smooth(:,end-avg_extra-1:end-1,:),2);
%    end_right = nanmean(Bxyz_fit_smooth(:,end-avg_extra:end,:),2)+slope_right*(avg_extra+1)/2;
%    Bxyz_extraX(:,SizeFactorForFFT_x*nbX/2+nbX/2,:) = end_right;
    for jj = 1:nbY
        slope_right(jj,1,:) = nanmean(nanmean(B_data(max(1,jj-avg_width):min(jj+avg_width,end),end-avg_extra:end,:)-B_data(max(1,jj-avg_width):min(jj+avg_width,end),end-avg_extra-1:end-1,:),1),2);
        end_value_right(jj,1,:) = nanmean(nanmean(B_data(max(1,jj-avg_width):min(jj+avg_width,end),end-avg_extra:end,:),1),2);
        end_value_right(jj,1,:) = end_value_right(jj,1,:)+slope_right(jj,1,:)*avg_extra/2;
        for ii = SizeFactorForFFT_x*nbX/2+nbX/2+1:SizeFactorForFFT_x*nbX
            for kk = 1:7
                if end_value_right(jj,1,kk)*slope_right(jj,1,kk) > 0 
                    slope_right(jj,1,kk) = -slope_right(jj,1,kk);
                end
                %Bxyz_extraX(jj,ii,kk) = Bxyz_extraX(jj,ii-1,kk)+slope_right(jj,1,kk);
                B_data_extraX(jj,ii,kk) = end_value_right(jj,1,kk)+slope_right(jj,1,kk)*(ii-(SizeFactorForFFT_x*nbX/2+nbX/2+1));
                if B_data_extraX(jj,ii,kk)*B_data_extraX(jj,ii-1,kk) <= 0 || padded == 1
                    B_data_extraX(jj,ii,kk) = 0;   % set to 0 if change sign
                end
            end
        end
        
    end
%     slope_left = nanmean(Bxyz_fit_smooth(:,1:1+avg_extra,:)-Bxyz_fit_smooth(:,2:2+avg_extra,:),2); 
%     end_left = nanmean(Bxyz_fit_smooth(:,1:1+avg_extra,:),2)+slope_left*(avg_extra+1)/2;
%     Bxyz_extraX(:,SizeFactorForFFT_x*nbX/2-nbX/2+1,:) = end_left;
    for jj = 1:nbY
        slope_left(jj,1,:) = nanmean(nanmean(B_data(max(1,jj-avg_width):min(jj+avg_width,end),1:1+avg_extra,:)-B_data(max(1,jj-avg_width):min(jj+avg_width,end),2:2+avg_extra,:),1),2);
        end_value_left(jj,1,:) = nanmean(nanmean(B_data(max(1,jj-avg_width):min(jj+avg_width,end),1:1+avg_extra,:),1),2);
        end_value_left(jj,1,:) = end_value_left(jj,1,:)-slope_left(jj,1,:)*avg_extra/2;
        for ii = SizeFactorForFFT_x*nbX/2-nbX/2:-1:1
            for kk = 1:7 
                if end_value_left(jj,1,kk)*slope_left(jj,1,kk) > 0 
                    slope_left(jj,1,kk) = -slope_left(jj,1,kk);
                end
                %Bxyz_extraX(jj,ii,kk) = Bxyz_extraX(jj,ii+1,kk)+slope_left(jj,1,kk);
                B_data_extraX(jj,ii,kk) = end_value_left(jj,1,kk)+slope_left(jj,1,kk)*((SizeFactorForFFT_x*nbX/2-nbX/2)-ii);
                if B_data_extraX(jj,ii,kk)*B_data_extraX(jj,ii+1,kk) <= 0 || padded == 1
                    B_data_extraX(jj,ii,kk) = 0;   % set to 0 if change sign
                end
            end
        end
    end
    figure(30); 
    imagesc(B_data_extraX(:,:,3)*1e2,[-range_B,range_B]); 
    colormap(colour); axis equal;
end


% Extrapolate in y
if SizeFactorForFFT_y == 1
    B_data_extra(:,:,:) = B_data_extraX(:,:,:);
else    
    B_data_extra(SizeFactorForFFT_y*nbY/2-nbY/2+1:SizeFactorForFFT_y*nbY/2+nbY/2,:,:) = B_data_extraX(:,:,:);
    for ii = SizeFactorForFFT_x*nbX/2-nbX/2+1:SizeFactorForFFT_x*nbX/2+nbX/2
        slope_bottom(1,ii,:) = nanmean(nanmean(B_data_extraX(end-avg_extra:end,max(1,ii-avg_width):min(ii+avg_width,end),:)-B_data_extraX(end-avg_extra-1:end-1,max(1,ii-avg_width):min(ii+avg_width,end),:),1),2);
        end_value_bottom(1,ii,:) = nanmean(nanmean(B_data_extraX(end-avg_extra:end,max(1,ii-avg_width):min(ii+avg_width,end),:),1),2);
        end_value_bottom(1,ii,:) = end_value_bottom(1,ii,:)+slope_bottom(1,ii,:)*avg_extra/2;
        for jj = SizeFactorForFFT_y*nbY/2+nbY/2+1:SizeFactorForFFT_y*nbY
            for kk = 1:7
                if end_value_bottom(1,ii,kk)*slope_bottom(1,ii,kk) > 0 
                   slope_bottom(1,ii,kk) = -slope_bottom(1,ii,kk);
                end
                %Bxyz_extra(jj,ii,kk) = Bxyz_extra(jj-1,ii,kk)+slope_bottom(1,ii,kk);
                B_data_extra(jj,ii,kk) = end_value_bottom(1,ii,kk)+slope_bottom(1,ii,kk)*(jj-(SizeFactorForFFT_y*nbY/2+nbY/2+1));
                if B_data_extra(jj,ii,kk)*B_data_extra(jj-1,ii,kk) <= 0 || padded == 1
                    B_data_extra(jj,ii,kk) = 0;   % set to 0 if change sign
                end
            end
        end
        
    end
    
    for ii = SizeFactorForFFT_x*nbX/2-nbX/2+1:SizeFactorForFFT_x*nbX/2+nbX/2
        slope_top(1,ii,:) = nanmean(nanmean(B_data_extraX(1:1+avg_extra,max(1,ii-avg_width):min(ii+avg_width,end),:)-B_data_extraX(2:2+avg_extra,max(1,ii-avg_width):min(ii+avg_width,end),:),1),2);
        end_value_top(1,ii,:) = nanmean(nanmean(B_data_extraX(1:1+avg_extra,max(1,ii-avg_width):min(ii+avg_width,end),:),1),2);
        end_value_top(1,ii,:) = end_value_top(1,ii,:)-slope_top(1,ii,:)*avg_extra/2;
        for jj = SizeFactorForFFT_y*nbY/2-nbY/2:-1:1
            for kk = 1:7
                if end_value_top(1,ii,kk)*slope_top(1,ii,kk) > 0 
                    slope_top(1,ii,kk) = -slope_top(1,ii,kk);
                end
                %Bxyz_extra(jj,ii,kk) = Bxyz_extra(jj+1,ii,kk)+slope_top(1,ii,kk);
                B_data_extra(jj,ii,kk) = end_value_top(1,ii,kk)+slope_top(1,ii,kk)*((SizeFactorForFFT_y*nbY/2-nbY/2)-jj);
                if B_data_extra(jj,ii,kk)*B_data_extra(jj+1,ii,kk) <= 0 || padded == 1
                    B_data_extra(jj,ii,kk) = 0;   % set to 0 if change sign
                end
            end
        end
    end
    B_data_extra(SizeFactorForFFT_y*nbY/2-nbY/2+1:SizeFactorForFFT_y*nbY/2+nbY/2,SizeFactorForFFT_x*nbX/2-nbX/2+1:SizeFactorForFFT_x*nbX/2+nbX/2,:) = Bxyz_fit(:,:,:);
    figure(31); 
    imagesc(B_data_extra(:,:,3)*1e2,[-range_B,range_B]); 
    colormap(colour); axis equal;
end

%% FFT to check Bxyz
%%%%%%%%%%%%%%%%%%%%%

clear bnv;
% Move to Fourier space
bx = fftshift(fft2(B_data_extra(:,:,1)*1e-4,SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX));  
by = fftshift(fft2(B_data_extra(:,:,2)*1e-4,SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX));  
bz = fftshift(fft2(B_data_extra(:,:,3)*1e-4,SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX));
for ii = 1:4
    bnv(:,:,ii) = fftshift(fft2(B_data_extra(:,:,ii+3)*1e-4,SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX));
end
[nbKy,nbKx] = size(bx);
pixelsize_kx = 1/(pixelsize_x*nbKx)*2*pi;
pixelsize_ky = 1/(pixelsize_y*nbKy)*2*pi;
kx_vec = -pixelsize_kx*(nbKx/2):pixelsize_kx:pixelsize_kx*(nbKx/2-1);    
ky_vec = -pixelsize_ky*(nbKy/2):pixelsize_ky:pixelsize_ky*(nbKy/2-1); 
[kx,ky] = meshgrid(-kx_vec,ky_vec); % the minus sign is needed to give consistent results, assuming the NV plane is above the sources
k = sqrt(kx.^2+ky.^2);

% Transformation matrices
bx_rec = NV_above_or_below*1i*kx./k.*bz;                      	
by_rec = NV_above_or_below*1i*ky./k.*bz;                         
bz_rec_from_planar = -NV_above_or_below*(1i*kx.*bx+1i*ky.*by)./k; 
by_rec_from_bx = ky.*bx./kx;                    
bx_rec_from_by = kx.*by./ky;                    
bz_rec = bz;
bx_rec_temp = bx_rec;

% Regularise
bx_rec(isnan(bx_rec)|isinf(bx_rec)) = 0;
by_rec(isnan(by_rec)|isinf(by_rec)) = 0;
bz_rec_from_planar(isnan(bz_rec_from_planar)|isinf(bz_rec_from_planar)) = 0;
by_rec_from_bx(isnan(by_rec_from_bx)|isinf(by_rec_from_bx)) = 0;
bx_rec_from_by(isnan(bx_rec_from_by)|isinf(bx_rec_from_by)) = 0;

% Back to real space
Bx_rec = real(ifft2(ifftshift(bx_rec)))*1e4; Bxyz_rec(:,:,1) = Bx_rec(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
By_rec = real(ifft2(ifftshift(by_rec)))*1e4; Bxyz_rec(:,:,2) = By_rec(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
Bz_rec = real(ifft2(ifftshift(bz_rec)))*1e4; Bxyz_rec(:,:,3) = Bz_rec(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
Bxyz_rec(:,:,1) = Bxyz_rec(:,:,1)-mean(mean(Bxyz_rec(y_subplane,x_subplane,1)));  
Bxyz_rec(:,:,2) = Bxyz_rec(:,:,2)-mean(mean(Bxyz_rec(y_subplane,x_subplane,2)));  
Bx_rec_from_By = real(ifft2(ifftshift(bx_rec_from_by)))*1e4; Bxyz_rec_from_planar(:,:,1) = Bx_rec_from_By(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
By_rec_from_Bx = real(ifft2(ifftshift(by_rec_from_bx)))*1e4; Bxyz_rec_from_planar(:,:,2) = By_rec_from_Bx(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
Bz_rec_from_planar = real(ifft2(ifftshift(bz_rec_from_planar)))*1e4; Bxyz_rec_from_planar(:,:,3) = Bz_rec_from_planar(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
Bxyz_fit_crop(:,:,:) = Bxyz_fit(1+crop:nbY-crop,1+crop:nbX-crop,:);

% Plot
figure(32)
PL_crop = PL(1+crop:nbY-crop,1+crop:nbX-crop); 
imagesc(PL_crop);
colorbar; colormap(gray); axis equal;
axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
set(gca,'FontSize',10); set(gcf,'units','points','position',[100,100,600,400]);
set(gca,'position',[0 0 1 1],'units','normalized'); axis off;
set(gca,'LooseInset',get(gca,'TightInset'));
set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
if border_free == 1
    hFigure = figure(32);
    set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
end
saveas(figure(32),strcat(Save_Path,'/PL_crop.png'));
datatosave = PL_crop; save(strcat(Save_Path, '/PL_crop.txt'), 'datatosave' , '-ascii');
[nbY_crop,nbX_crop] =  size(PL_crop);

figure(40)
subplot(3,3,1); imagesc(Bxyz_fit_crop(:,:,1)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_x (T) meas. '); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,2); imagesc(Bxyz_fit_crop(:,:,2)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_y (T) meas. '); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,3); imagesc(Bxyz_fit_crop(:,:,3)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_z (T) meas. '); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,4); imagesc(Bxyz_rec(:,:,1)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_x (T) rec. from B_z via curl(B)=0'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,5); imagesc(Bxyz_rec(:,:,2)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_y (T) rec. from B_z via curl(B)=0'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,6); imagesc(Bxyz_rec_from_planar(:,:,3)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_z (T) rec. from B_x and B_y via div(B)=0'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,7); imagesc(Bxyz_rec_from_planar(:,:,1)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_x (T) rec. from B_y via curl(B)=0'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
subplot(3,3,8); imagesc(Bxyz_rec_from_planar(:,:,2)*1e2,[-range_B,range_B]); colorbar; axis equal; title('B_y (T) rec. from B_x via curl(B)=0'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
colormap(colour); set(gca,'FontSize',10); set(gcf,'units','points','position',[10,10,1000,800]);
saveas(figure(40),strcat(Save_Path,'/Bxyz_reconstructed.png'));

if plot_individuals == 1
    
    for ii = 1:3
        figure(40+ii)
        if auto_colour_range == 1
            imagesc(Bxyz_rec(:,:,ii)*1e2);
        else
            imagesc(Bxyz_rec(:,:,ii)*1e2,[-range_B,range_B]);
        end
        colorbar; axis equal; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
        colormap(colour); set(gca,'FontSize',10); 
        set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
        if border_free == 1
            hFigure = figure(40+ii);
            set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
            set(gca,'LooseInset',get(gca,'TightInset'));
            set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
        end
    end

    saveas(figure(41),strcat(Save_Path,'\Bx_rec.png'));
    saveas(figure(42),strcat(Save_Path,'\By_rec.png'));
    saveas(figure(43),strcat(Save_Path,'\Bz_rec.png'));

    for ii = 1:3
        figure(43+ii) 
        if auto_colour_range == 1
            imagesc(Bxyz_fit_crop(:,:,ii)*1e2);
        else
            imagesc(Bxyz_fit_crop(:,:,ii)*1e2,[-range_B,range_B]);
        end
        colorbar; axis equal; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
        colormap(colour); set(gca,'FontSize',10); 
        set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
        if border_free == 1
            hFigure = figure(43+ii);
            set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
            set(gca,'LooseInset',get(gca,'TightInset'));
            set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
        end
    end

    saveas(figure(44),strcat(Save_Path,'\Bx_crop.png'));
    saveas(figure(45),strcat(Save_Path,'\By_crop.png'));
    saveas(figure(46),strcat(Save_Path,'\Bz_crop.png'));
    
    figure(47)
    if auto_colour_range == 1
        imagesc(Bxyz_rec_from_planar(:,:,3)*1e2);
    else
        imagesc(Bxyz_rec_from_planar(:,:,3)*1e2,[-range_B,range_B]);
    end
    colorbar; axis equal; title(title_vec(ii)); set(gca,'xtick',[]); set(gca,'ytick',[]);
    colormap(colour); set(gca,'FontSize',10); 
    set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
    if border_free == 1
        hFigure = figure(47);
        set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
        set(gca,'LooseInset',get(gca,'TightInset'));
        set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
    end
    saveas(figure(47),strcat(Save_Path,'\Bz_rec.png'));

end

%% FFT to get Jxy
%%%%%%%%%%%%%%%%%%
if calc_J
    % Compute exponential factor 
    exp_fac = exp(-k*standoff);
    if NV_layer_thickness ~= 0
        exp_fac = exp_fac.*sinh(k*NV_layer_thickness/2)./(k*NV_layer_thickness/2);
    end
    exp_fac(isnan(exp_fac)) = 1;

    % Define Hanning filter to prevent noise amplification at frequencies higher than the spatial resolution 
    k_cut = 2*pi/standoff;
    Hanning = 0.5*(1 + cos(pi*k/k_cut));
    Hanning(k>k_cut) = 0;

    % Transformation matrices
    jy_from_bz = Hanning*2/mu0./exp_fac .* bz.*(+1i*kx./k); 
    jx_from_bz = Hanning*2/mu0./exp_fac .* bz.*(-1i*ky./k); 
    jx_from_by = Hanning*2/mu0./exp_fac .* by.*(-NV_above_or_below);        
    jy_from_bx = Hanning*2/mu0./exp_fac .* bx.*(+NV_above_or_below);        

    % Regularise
    jy_from_bz(isnan(jy_from_bz)|isinf(jy_from_bz)) = 0;
    jx_from_bz(isnan(jx_from_bz)|isinf(jx_from_bz)) = 0;
    jx_from_by(isnan(jx_from_by)|isinf(jx_from_by)) = 0;
    jy_from_bx(isnan(jy_from_bx)|isinf(jy_from_bx)) = 0;

    % Back to real space
    Jx_from_By = real(ifft2(ifftshift(jx_from_by))); Jxy_from_Bxy(:,:,1) = Jx_from_By(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
    Jy_from_Bx = real(ifft2(ifftshift(jy_from_bx))); Jxy_from_Bxy(:,:,2) = Jy_from_Bx(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
    Jx_from_Bz = real(ifft2(ifftshift(jx_from_bz))); Jxy_from_Bz(:,:,1) = Jx_from_Bz(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
    Jy_from_Bz = real(ifft2(ifftshift(jy_from_bz))); Jxy_from_Bz(:,:,2) = Jy_from_Bz(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop);
    Jxy_from_Bz(:,:,1) = Jxy_from_Bz(:,:,1)-mean(mean(Jxy_from_Bz(y_subplane,x_subplane,1))); 
    Jxy_from_Bz(:,:,2) = Jxy_from_Bz(:,:,2)-mean(mean(Jxy_from_Bz(y_subplane,x_subplane,2)));
    Jxy_diff = Jxy_from_Bz-Jxy_from_Bxy; 
    Jxy_below_NV = Jxy_diff/2;
    Jxy_above_NV = Jxy_from_Bz-Jxy_below_NV;
    Jxy_from_Bz(:,:,3) = sqrt(Jxy_from_Bz(:,:,1).^2+Jxy_from_Bz(:,:,2).^2); 
    Jxy_from_Bxy(:,:,3) = sqrt(Jxy_from_Bxy(:,:,1).^2+Jxy_from_Bxy(:,:,2).^2); 
    Jxy_diff(:,:,3) = sqrt(Jxy_diff(:,:,1).^2+Jxy_diff(:,:,2).^2); 
    Jxy_above_NV(:,:,3) = sqrt(Jxy_above_NV(:,:,1).^2+Jxy_above_NV(:,:,2).^2); 
    Jxy_below_NV(:,:,3) = sqrt(Jxy_below_NV(:,:,1).^2+Jxy_below_NV(:,:,2).^2); 

    % Get J from BNV
    for ii = 1:4
        jx_from_bnv_temp = Hanning*2/mu0./exp_fac.*bnv(:,:,ii).*ky./(-NV_above_or_below*uNV(ii,1)*kx-NV_above_or_below*uNV(ii,2)*kx+uNV(ii,3)*1i*k); 
        jy_from_bnv_temp = Hanning*2/mu0./exp_fac.*bnv(:,:,ii).*kx./(+NV_above_or_below*uNV(ii,1)*kx+NV_above_or_below*uNV(ii,2)*kx-uNV(ii,3)*1i*k);
        jx_from_bnv_temp(isnan(jx_from_bnv_temp)|isinf(jx_from_bnv_temp)) = 0;
        jy_from_bnv_temp(isnan(jy_from_bnv_temp)|isinf(jy_from_bnv_temp)) = 0;
        Jx_from_BNV_temp = real(ifft2(ifftshift(jx_from_bnv_temp))); 
        Jy_from_BNV_temp = real(ifft2(ifftshift(jy_from_bnv_temp)));
        Jxy_from_BNV(:,:,ii,1) = Jx_from_BNV_temp(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
        Jxy_from_BNV(:,:,ii,2) = Jy_from_BNV_temp(1+SizeFactorForFFT_y*nbY/2-nbY/2+crop:SizeFactorForFFT_y*nbY/2+nbY/2-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
        Jxy_from_BNV(:,:,ii,1) = Jxy_from_BNV(:,:,ii,1)-mean(mean(Jxy_from_BNV(y_subplane,x_subplane,ii,1)));
        Jxy_from_BNV(:,:,ii,2) = Jxy_from_BNV(:,:,ii,2)-mean(mean(Jxy_from_BNV(y_subplane,x_subplane,ii,2)));
        Jxy_from_BNV(:,:,ii,3) = sqrt(Jxy_from_BNV(:,:,ii,1).^2+Jxy_from_BNV(:,:,ii,2).^2);
    end

    figure(50)
    subplot(2,3,1); imagesc(Jxy_from_Bz(:,:,1),[-range_J,range_J]); colorbar; axis equal; title('J_x from B_z (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,4); imagesc(Jxy_from_Bz(:,:,2),[-range_J,range_J]); colorbar; axis equal; title('J_y from B_z (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,2); imagesc(Jxy_from_Bxy(:,:,1),[-range_J,range_J]); colorbar; axis equal; title('J_x from B_{xy} (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,5); imagesc(Jxy_from_Bxy(:,:,2),[-range_J,range_J]); colorbar; axis equal; title('J_y from B_{xy} (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,3); imagesc(Jxy_diff(:,:,1),[-range_J,range_J]); colorbar; axis equal; title('\Delta J_x (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,6); imagesc(Jxy_diff(:,:,2),[-range_J,range_J]); colorbar; axis equal; title('\Delta J_y (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    set(gca,'FontSize',10); set(gcf,'units','points','position',[60,20,860,400]); colormap(colour);
    saveas(figure(50),strcat(Save_Path,'/Jxy.png'));

    if plot_individuals == 1
        Jy_to_plot(:,:,1) = Jxy_tot(:,:,1); Jy_to_plot(:,:,2) = Jxy_wire(:,:,1); Jy_to_plot(:,:,3) = Jxy_NV(:,:,1); 
        Jy_to_plot(:,:,4) = Jxy_tot(:,:,2); Jy_to_plot(:,:,5) = Jxy_wire(:,:,2); Jy_to_plot(:,:,6) = Jxy_NV(:,:,2);
        for ii = 1:6
            figure(50+ii)
            if auto_colour_range == 1
                imagesc(Jy_to_plot(:,:,ii));
            else
                imagesc(Jy_to_plot(:,:,ii),[-range_J,range_J]);
            end
            colorbar; axis equal; set(gca,'xtick',[]); set(gca,'ytick',[]);
            colormap(colour); set(gca,'FontSize',10); 
            set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
            if border_free == 1
                hFigure = figure(50+ii);
                set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
                set(gca,'LooseInset',get(gca,'TightInset'));
                set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
            end
        end
        saveas(figure(51),strcat(Save_Path,'\Jx_tot.png'));
        saveas(figure(52),strcat(Save_Path,'\Jx_wire.png'));
        saveas(figure(53),strcat(Save_Path,'\Jx_NV.png'));
        saveas(figure(54),strcat(Save_Path,'\Jy_tot.png'));
        saveas(figure(55),strcat(Save_Path,'\Jy_wire.png'));
        saveas(figure(56),strcat(Save_Path,'\Jy_NV.png'));
    end

    figure(65)
    for ii = 1:4
        subplot(3,4,ii); imagesc(Jxy_from_BNV(:,:,ii,1),[-range_J,range_J]); 
        colormap(redblue); colorbar; axis equal; title('J_x from B_{NV} (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(3,4,ii+4); imagesc(Jxy_from_BNV(:,:,ii,2),[-range_J,range_J]); 
        colormap(redblue); colorbar; axis equal; title('J_y from B_{NV} (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
        subplot(3,4,ii+8); imagesc(Jxy_from_BNV(:,:,ii,3),[0,range_J]); 
        colormap(jet); colorbar; axis equal; title('|J| from B_{NV} (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    end
    set(gca,'FontSize',10); set(gcf,'units','points','position',[60,20,1200,800]); 
    saveas(figure(65),strcat(Save_Path,'/Jnorm_from_BNV.png'));

    figure(58)
    subplot(2,3,1); imagesc(Jxy_from_Bz(:,:,1),[-range_J,range_J]); colorbar; axis equal; title('total J_x (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,4); imagesc(Jxy_from_Bz(:,:,2),[-range_J,range_J]); colorbar; axis equal; title('total J_y (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,2); imagesc(Jxy_above_NV(:,:,1),[-range_J,range_J]); colorbar; axis equal; title('J_x above NV (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,5); imagesc(Jxy_above_NV(:,:,2),[-range_J,range_J]); colorbar; axis equal; title('J_y above NV (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,3); imagesc(Jxy_below_NV(:,:,1),[-range_J,range_J]); colorbar; axis equal; title('J_x below NV (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    subplot(2,3,6); imagesc(Jxy_below_NV(:,:,2),[-range_J,range_J]); colorbar; axis equal; title('J_y below NV (A/m)'); axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    set(gca,'FontSize',10); set(gcf,'units','points','position',[60,20,860,400]); colormap(colour);
    saveas(figure(58),strcat(Save_Path,'/Jxy_above_below_NV.png'));

    if plot_individuals == 1
        Jy_to_plot(:,:,1) = Jxy_above(:,:,1); Jy_to_plot(:,:,2) = Jxy_below(:,:,1);
        Jy_to_plot(:,:,3) = Jxy_above(:,:,2); Jy_to_plot(:,:,4) = Jxy_below(:,:,2);
        for ii = 1:4
            figure(60+ii)
            if auto_colour_range == 1
                imagesc(Jy_to_plot(:,:,ii));
            else
                imagesc(Jy_to_plot(:,:,ii),[-range_J,range_J]);
            end
            colorbar; axis equal; set(gca,'xtick',[]); set(gca,'ytick',[]);
            colormap(colour); set(gca,'FontSize',10); 
            set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
            if border_free == 1
                hFigure = figure(60+ii);
                set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
                set(gca,'LooseInset',get(gca,'TightInset'));
                set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
            end
        end
        saveas(figure(61),strcat(Save_Path,'\Jx_above.png'));
        saveas(figure(62),strcat(Save_Path,'\Jx_below.png'));
        saveas(figure(63),strcat(Save_Path,'\Jy_above.png'));
        saveas(figure(64),strcat(Save_Path,'\Jy_below.png'));
    end
end
%% Plot Jnorm 
%%%%%%%%%%%%%%
if calc_J
    kk = 0; clear x_arrow y_arrow Jx_arrow Jy_arrow
    for ii = 7:4:nbY_crop-2
        for jj = 7:4:nbX_crop-3
            if Jxy_from_Bxy(ii,jj,3) > 10 && Jxy_from_Bxy(ii,jj,3) < 120
                kk = kk+1; x_arrow(kk) = jj; y_arrow(kk) = ii; 
                Jx_arrow(1,kk) = Jxy_from_Bz(ii,jj,1); Jy_arrow(1,kk) = -Jxy_from_Bz(ii,jj,2);
                Jx_arrow(2,kk) = Jxy_from_Bxy(ii,jj,1); Jy_arrow(2,kk) = -Jxy_from_Bxy(ii,jj,2);
                Jx_arrow(3,kk) = Jxy_diff(ii,jj,1); Jy_arrow(3,kk) = -Jxy_diff(ii,jj,2);
                Jx_arrow(4,kk) = Jxy_above_NV(ii,jj,1); Jy_arrow(4,kk) = -Jxy_above_NV(ii,jj,2);
                Jx_arrow(5,kk) = Jxy_below_NV(ii,jj,1); Jy_arrow(5,kk) = -Jxy_below_NV(ii,jj,2);
            end
        end
    end
    if kk == 0
        x_arrow = 10; y_arrow = 10; Jx_arrow(1:5,1) = zeros(5,1); Jy_arrow(1:5,1) = zeros(5,1);
    end

    ArrowScaleFactor = 4*[0.5,0.5,0.5,0.5,0.5];
    figure(65)
    subplot(2,3,1); imagesc(Jxy_from_Bz(:,:,3),[0,range_J]); colorbar; title('|J| from B_z (A/m)'); axis equal; axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    hold on; quiver(x_arrow,y_arrow,Jx_arrow(1,:),Jy_arrow(1,:),'color',[1 1 1],'AutoScaleFactor',ArrowScaleFactor(1),'AutoScale','on');
    subplot(2,3,2); imagesc(Jxy_from_Bxy(:,:,3),[0,range_J]); colorbar; title('|J| from B_{xy} (A/m)'); axis equal; axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    hold on; quiver(x_arrow,y_arrow,Jx_arrow(2,:),Jy_arrow(2,:),'color',[1 1 1],'AutoScaleFactor',ArrowScaleFactor(2),'LineWidth',1,'AutoScale','on');
    subplot(2,3,3); imagesc(Jxy_diff(:,:,3),[0,range_J]); colorbar; title('\Delta |J| (A/m)'); axis equal; axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    hold on; quiver(x_arrow,y_arrow,Jx_arrow(3,:),Jy_arrow(3,:),'color',[1 1 1],'AutoScaleFactor',ArrowScaleFactor(3),'AutoScale','on');
    subplot(2,3,5); imagesc(Jxy_above_NV(:,:,3),[0,range_J]); colorbar; title('|J| above NV (A/m)'); axis equal; axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    hold on; quiver(x_arrow,y_arrow,Jx_arrow(4,:),Jy_arrow(4,:),'color',[1 1 1],'AutoScaleFactor',ArrowScaleFactor(4),'AutoScale','on');
    subplot(2,3,6); imagesc(Jxy_below_NV(:,:,3),[0,range_J]); colorbar; title('|J| below NV (A/m)'); axis equal; axis tight; set(gca,'xtick',[]); set(gca,'ytick',[]);
    hold on; quiver(x_arrow,y_arrow,Jx_arrow(5,:),Jy_arrow(5,:),'color',[1 1 1],'AutoScaleFactor',ArrowScaleFactor(5),'AutoScale','on');
    set(gca,'FontSize',10); set(gcf,'units','points','position',[60,20,860,400]); colormap(jet); %set(colorbar,'TickLength',0.04,'FontSize',9,'Position',[0.92,0.18,0.008,0.18]);
    saveas(figure(65),strcat(Save_Path,'/Jnorm.emf'));
    saveas(figure(65),strcat(Save_Path,'/Jnorm.png'));

    Jnorm_to_plot(:,:,1) = Jxy_from_Bz(:,:,3);
    Jnorm_to_plot(:,:,2) = Jxy_from_Bxy(:,:,3); Jnorm_to_plot(:,:,3) = Jxy_diff(:,:,3);
    Jnorm_to_plot(:,:,4) = Jxy_above_NV(:,:,3); Jnorm_to_plot(:,:,5) = Jxy_below_NV(:,:,3);

    if plot_individuals == 1
        for ii = 1:5
            figure(65+ii)
            if auto_colour_range == 1
                imagesc(Jnorm_to_plot(:,:,ii));
            else
                imagesc(Jnorm_to_plot(:,:,ii),[0,range_J]);
            end
            colorbar; axis equal; set(gca,'xtick',[]); set(gca,'ytick',[]);
            colormap(jet); set(gca,'FontSize',10); 
            set(gcf,'units','points','position',[50,5,RealSizePlot(2),RealSizePlot(1)]);
            if border_free == 1
                hFigure = figure(65+ii);
                set(gca,'position',[0 0 1 1],'units','normalized'); axis off; axis tight;
                set(gca,'LooseInset',get(gca,'TightInset'));
                set(hFigure, 'MenuBar', 'none'); set(hFigure, 'ToolBar', 'none');
            end
            hold on; quiver(x_arrow,y_arrow,Jx_arrow(ii,:),Jy_arrow(ii,:),'color',[0 0 0],'AutoScaleFactor',ArrowScaleFactor(ii));
        end

    saveas(figure(66),strcat(Save_Path,'\Jnorm_from_Bz.emf'));
    saveas(figure(67),strcat(Save_Path,'\Jnorm_from_Bxy.emf'));
    saveas(figure(68),strcat(Save_Path,'\Jnorm_diff.emf'));
    saveas(figure(69),strcat(Save_Path,'\Jnorm_above_NV.emf'));
    saveas(figure(70),strcat(Save_Path,'\Jnorm_below_NV.emf'));

    figure(71)
    datatosave = plot_image(Jxy_from_Bz(:,:,3),'|J| from B_z (A/m)',0,1,[0,0],Full_ROI,ROI_square,ROI_centre,ROI_radius);
    saveas(figure(71),strcat(Save_Path,'/Jnorm_from_Bz.png'));

    end

    datatosave = Jxy_from_Bz(:,:,1);
    save(strcat(Save_Path, '/Jx_from_Bz.txt'), 'datatosave' , '-ascii');
    datatosave = Jxy_from_Bz(:,:,2);
    save(strcat(Save_Path, '/Jy_from_Bz.txt'), 'datatosave' , '-ascii');
    datatosave = Jxy_from_Bz(:,:,3);
    save(strcat(Save_Path, '/Jnorm_from_Bz.txt'), 'datatosave' , '-ascii');
    datatosave = Jxy_from_Bxy(:,:,1);
    save(strcat(Save_Path, '/Jx_from_Bxy.txt'), 'datatosave' , '-ascii');
    datatosave = Jxy_from_Bxy(:,:,2);
    save(strcat(Save_Path, '/Jy_from_Bxy.txt'), 'datatosave' , '-ascii');
    datatosave = Jxy_from_Bxy(:,:,3);
    save(strcat(Save_Path, '/Jnorm_from_Bxy.txt'), 'datatosave' , '-ascii');
end
%% Plot with circular mask

if plot_circular_mask == 1
    figure(71)
    datatosave = plot_image(Bxyz_fit_crop(:,:,3)*1e2,'B_z (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(71),strcat(Save_Path,'/Bz_mask.png'));
    save(strcat(Save_Path, '/Bz_mask.txt'), 'datatosave' , '-ascii');
    figure(72)
    datatosave = plot_image(Jxy_from_Bz(:,:,3),'|J| from B_z (A/m)',2,0,0,[0,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(72),strcat(Save_Path,'/Jnorm_tot_mask.png'));
    save(strcat(Save_Path, '/Jnorm_from_Bz_mask.txt'), 'datatosave' , '-ascii');
    figure(73)
    datatosave = plot_image(C_mat(:,:,1),'Contrast',2,0,0,[0.004,0.012],2,[0,0],ROI_centre,ROI_radius);
    saveas(figure(73),strcat(Save_Path,'/contrast_mask.png'));
    save(strcat(Save_Path, '/contrast_mask.txt'), 'datatosave' , '-ascii');
elseif plot_circular_mask == 2
    figure(71)
    datatosave = plot_image(Bxyz_fit_crop(:,:,1)*1e2,'B_x (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(71),strcat(Save_Path,'/Bx_mask.png'));
    save(strcat(Save_Path, '/Bx_mask.txt'), 'datatosave' , '-ascii');
    figure(72)
    datatosave = plot_image(Bxyz_fit_crop(:,:,2)*1e2,'B_y (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(72),strcat(Save_Path,'/By_mask.png'));
    save(strcat(Save_Path, '/By_mask.txt'), 'datatosave' , '-ascii');
    figure(73)
    datatosave = plot_image(Bxyz_fit_crop(:,:,3)*1e2,'B_z (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(73),strcat(Save_Path,'/Bz_mask.png'));
    save(strcat(Save_Path, '/Bz_mask.txt'), 'datatosave' , '-ascii');
    figure(74)
    datatosave = plot_image(Bxyz_rec(:,:,1)*1e2,'B_x rec. (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(74),strcat(Save_Path,'/Bx_rec_mask.png'));
    save(strcat(Save_Path, '/Bx_rec_mask.txt'), 'datatosave' , '-ascii');
    figure(75)
    datatosave = plot_image(Bxyz_rec(:,:,2)*1e2,'B_y rec. (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(75),strcat(Save_Path,'/By_rec_mask.png'));
    save(strcat(Save_Path, '/By_rec_mask.txt'), 'datatosave' , '-ascii');
    figure(76)
    datatosave = plot_image(Bxyz_rec_from_planar(:,:,3)*1e2,'B_z rec. (uT)',1,0,0,[-range_B,range_B],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(76),strcat(Save_Path,'/Bz_rec_mask.png'));
    save(strcat(Save_Path, '/Bz_rec_mask.txt'), 'datatosave' , '-ascii');
%     figure(77)
%     datatosave = plot_image(Jxy_tot(:,:,1),'J_x total',1,0,[-range_J,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
%     saveas(figure(77),strcat(Save_Path,'/Jx_tot_mask.png'));
%     save(strcat(Save_Path, '/Jx_tot_mask.txt'), 'datatosave' , '-ascii');
%     figure(78)
%     datatosave = plot_image(Jxy_above(:,:,1),'J_x above',1,0,[-range_J,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
%     saveas(figure(78),strcat(Save_Path,'/Jx_above_mask.png'));
%     save(strcat(Save_Path, '/Jx_above_mask.txt'), 'datatosave' , '-ascii');
%     figure(79)
%     datatosave = plot_image(Jxy_below(:,:,1),'J_x below',1,0,[-range_J,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
%     saveas(figure(79),strcat(Save_Path,'/Jx_below_mask.png'));
%     save(strcat(Save_Path, '/Jx_below_mask.txt'), 'datatosave' , '-ascii');
    figure(77)
    datatosave = plot_image(Jxy_from_Bz(:,:,3),'J_x from B_z',2,0,1,[0,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(77),strcat(Save_Path,'/Jx_from_Bz_mask.png'));
    save(strcat(Save_Path, '/Jx_from_Bz_mask.txt'), 'datatosave' , '-ascii');
    figure(78)
    datatosave = plot_image(Jxy_above_NV(:,:,3),'J_x above NV',2,0,1,[0,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(78),strcat(Save_Path,'/Jx_above_NV_mask.png'));
    save(strcat(Save_Path, '/Jx_above_NV_mask.txt'), 'datatosave' , '-ascii');
    figure(79)
    datatosave = plot_image(Jxy_below_NV(:,:,3),'J_x below NV',2,0,1,[0,range_J],2,[0,0],ROI_centre-[crop,crop],ROI_radius);
    saveas(figure(79),strcat(Save_Path,'/Jx_below_NV_mask.png'));
    save(strcat(Save_Path, '/Jx_below_NV_mask.txt'), 'datatosave' , '-ascii');
end

%% Plot linecuts %%
%%%%%%%%%%%%%%%%%%%
if plot_linecuts
    width = 15e-6;          % wire width in m
    thickness = 10e-9;      % wire thickness in m
    I = 6e-3;               % I in A;
    FWHM_OpticalRes = 1000e-9;
    x_offset = 8e-6;
    y_offset = -0e-6;

    x_vec = -floor(nbX_crop/2)*pixelsize_x:pixelsize_x:(ceil(nbX_crop/2)-1)*pixelsize_x;
    y_vec = -floor(nbY_crop/2)*pixelsize_y:pixelsize_y:(ceil(nbY_crop/2)-1)*pixelsize_y;
    sigma = FWHM_OpticalRes/(2*sqrt(2*log(2)));
    gaussFilter = exp(-x_vec .^ 2 / (2 * sigma ^ 2));
    gaussFilter = gaussFilter / sum (gaussFilter);

    mu0 = 4e-7*pi; dz = 1e-9;
    z_vec = 0:dz:thickness; nbZ = length(z_vec);
    By_analytic = 0*x_vec; Bz_analytic = 0*x_vec;
    for ii = 1:nbZ
        zp = standoff+z_vec(ii);
        By_analytic(1,:) = By_analytic(1,:)+mu0*I/(2*pi*width*nbZ)*(atan((width-2*x_vec)/(2*zp))+atan((width+2*x_vec)/(2*zp)));
        Bz_analytic(1,:) = Bz_analytic(1,:)+mu0*I/(4*pi*width*nbZ)*log(((width-2*x_vec).^2+(2*zp)^2)./((width+2*x_vec).^2+(2*zp)^2));
    end
    % By_analytic(1,:) = mu0*I/(2*pi*width)*(atan((width-2*x_vec)/(2*zp))+atan((width+2*x_vec)/(2*zp)));
    % Bz_analytic(1,:) = -mu0*I/(4*pi*width)*log(((width-2*x_vec).^2+(2*zp)^2)./((width+2*x_vec).^2+(2*zp)^2));
    By_conv(1,:) = 1e6*conv(By_analytic(1,:),gaussFilter,'same');
    Bz_conv(1,:) = 1e6*conv(Bz_analytic(1,:),gaussFilter,'same');
    Bx_conv(1,:) = 0*x_vec;

    Bxyz_avg_hor(:,:,:) = nanmean(Bxyz_fit_crop(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1)*1e2;
    Bxyz_avg_ver(:,:,:) = nanmean(Bxyz_fit_crop(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2)*1e2;
    Bxyz_rec_hor(:,:,:) = nanmean(Bxyz_rec(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1)*1e2;
    Bxyz_rec_ver(:,:,:) = nanmean(Bxyz_rec(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2)*1e2;
    Bxyz_rec_from_planar_hor(:,:,:) = nanmean(Bxyz_rec_from_planar(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1)*1e2;
    Bxyz_rec_from_planar_ver(:,:,:) = nanmean(Bxyz_rec_from_planar(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2)*1e2; 
    Jxy_from_Bz_hor(:,:,:) = nanmean(Jxy_from_Bz(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1);
    Jxy_from_Bxy_hor(:,:,:) = nanmean(Jxy_from_Bxy(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1);
    Jxy_diff_hor(:,:,:) = nanmean(Jxy_diff(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1);
    Jxy_from_Bz_ver(:,:,:) = nanmean(Jxy_from_Bz(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2);
    Jxy_from_Bxy_ver(:,:,:) = nanmean(Jxy_from_Bxy(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2);
    Jxy_diff_ver(:,:,:) = nanmean(Jxy_diff(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2);
    Jxy_above_NV_hor(:,:,:) = nanmean(Jxy_above_NV(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1);
    Jxy_below_NV_hor(:,:,:) = nanmean(Jxy_below_NV(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1);
    Jxy_above_NV_ver(:,:,:) = nanmean(Jxy_above_NV(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2);
    Jxy_below_NV_ver(:,:,:) = nanmean(Jxy_below_NV(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2);
    PL_hor = nanmean(PL_crop(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:),1);
    PL_ver = nanmean(PL_crop(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width),2);
    PL_hor = PL_hor/max(PL_hor)*range_J;
    PL_ver = PL_ver/max(PL_ver)*range_J;
    if subtract_reference == 1
        if Linecut_subplane == 1
            Bxyz_avg_hor(:,:,:) = nanmean(Bxyz_fit_crop(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1)*1e2;
            Bxyz_avg_ver(:,:,:) = nanmean(Bxyz_fit_crop(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2)*1e2;
        else
            Bxyz_avg_hor(:,:,:) = nanmean(Bxyz_fit_crop(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1)*1e2;
            Bxyz_avg_ver(:,:,:) = nanmean(Bxyz_fit_crop(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2)*1e2;
            if plot_raw == 0
                Bxyz_avg_hor(:,:,:) = Bxyz_avg_hor(:,:,:)-nanmean(Bxyz_avg_hor(:,:,:));
                Bxyz_avg_ver(:,:,:) = Bxyz_avg_ver(:,:,:)-nanmean(Bxyz_avg_ver(:,:,:));
            end
        end
    end
    xvec = x_vec*1e6;
    X_Lim = [floor(xvec(1)) ceil(xvec(end))]-x_offset*1e6;
    Bx_hor_Lim = [floor(min(min(min(Bxyz_avg_hor(:,:,1)),min(By_conv(1,:))))) ceil(max(max(max(Bxyz_avg_hor(:,:,1)),max(By_conv(1,:)))))] ;
    By_hor_Lim = [floor(min(min(min(Bxyz_avg_hor(:,:,2)),min(Bx_conv(1,:))))) ceil(max(max(max(Bxyz_avg_hor(:,:,2)),max(Bx_conv(1,:)))))] ;
    Bz_hor_Lim = [floor(min(min(min(Bxyz_avg_hor(:,:,3)),min(Bz_conv(1,:))))) ceil(max(max(max(Bxyz_avg_hor(:,:,3)),max(Bz_conv(1,:)))))] ;
    Bx_ver_Lim = [floor(min(min(min(Bxyz_avg_ver(:,:,1)),min(Bx_conv(1,:))))) ceil(max(max(max(Bxyz_avg_ver(:,:,1)),max(Bx_conv(1,:)))))] ;
    By_ver_Lim = [floor(min(min(min(Bxyz_avg_ver(:,:,2)),min(By_conv(1,:))))) ceil(max(max(max(Bxyz_avg_ver(:,:,2)),max(By_conv(1,:)))))] ;
    Bz_ver_Lim = [floor(min(min(min(Bxyz_avg_ver(:,:,3)),min(Bz_conv(1,:))))) ceil(max(max(max(Bxyz_avg_ver(:,:,3)),max(Bz_conv(1,:)))))] ;

    figure(80);
    subplot(2,3,1);
    plot((x_vec-x_offset)*1e6,Bxyz_avg_hor(:,:,1),'r',xvec,By_conv(1,:),'r--',(x_vec-x_offset)*1e6,Bxyz_rec_hor(:,:,1),'r.-'); title('horizotal B_x (\muT)'); axis([X_Lim Bx_hor_Lim]);%ylim([-100,0]);
    legend('Meas.','Theory','Rec. from B_z');
    subplot(2,3,2);
    plot((x_vec-x_offset)*1e6,Bxyz_avg_hor(:,:,2),'b',xvec,Bx_conv(1,:),'b--'); title('horizontal B_y (\muT)'); axis([X_Lim By_hor_Lim]);%ylim([2700,2800]);
    subplot(2,3,3);
    plot((x_vec-x_offset)*1e6,Bxyz_avg_hor(:,:,3),'k',xvec,Bz_conv(1,:),'k--',(x_vec-x_offset)*1e6,Bxyz_rec_from_planar_hor(:,:,3),'k.-'); title('horizontal B_z (\muT)'); axis([X_Lim Bz_hor_Lim]);%ylim([-1750,-1700]);
    legend('Meas.','Theory','Rec. from B_x,B_y');
    subplot(2,3,4);
    plot((y_vec-y_offset)*1e6,Bxyz_avg_ver(:,:,1),'r',xvec,Bx_conv(1,:),'r--'); title('vertical B_x (\muT)'); axis([X_Lim Bx_ver_Lim]);
    subplot(2,3,5);
    plot((y_vec-y_offset)*1e6,Bxyz_avg_ver(:,:,2),'b',xvec,By_conv(1,:),'b--',(y_vec-y_offset)*1e6,Bxyz_rec_ver(:,:,2),'b.-'); title('vertical B_y (\muT)'); axis([X_Lim By_ver_Lim]);
    legend('Meas.','Theory','Rec. from B_z');
    subplot(2,3,6);
    plot((y_vec-y_offset)*1e6,Bxyz_avg_ver(:,:,3),'k',xvec,Bz_conv(1,:),'k--',(y_vec-y_offset)*1e6,Bxyz_rec_from_planar_ver(:,:,3),'k.-'); title('vertical B_z (\muT)'); axis([X_Lim Bz_ver_Lim]);
    legend('Meas.','Theory','Rec. from B_x,B_y');
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,10,600,400]);
    saveas(figure(80),strcat(Save_Path,'\Bxyz_linecuts.png'));

    figure(81);
    subplot(2,4,1);
    plot((x_vec-x_offset)*1e6,Jxy_from_Bz_hor(:,:,1),'r',(x_vec-x_offset)*1e6,Jxy_from_Bxy_hor(:,:,1),'r.-',(x_vec-x_offset)*1e6,Jxy_diff_hor(:,:,1),'r--'); title('horizontal J_x (\muT)'); xlim(X_Lim); %axis([X_Lim Jx_hor_Lim]);%ylim([-100,0]);
    legend('from B_z','from B_{xy}','difference');
    subplot(2,4,2);
    plot((x_vec-x_offset)*1e6,Jxy_from_Bz_hor(:,:,2),'b',(x_vec-x_offset)*1e6,Jxy_from_Bxy_hor(:,:,2),'b.-',(x_vec-x_offset)*1e6,Jxy_diff_hor(:,:,2),'b--',(x_vec-x_offset)*1e6,PL_hor,'k:'); title('horizontal J_y (\muT)'); xlim(X_Lim); %axis([X_Lim Jy_hor_Lim]);%ylim([2700,2800]);
    subplot(2,4,5);
    plot((y_vec-y_offset)*1e6,Jxy_from_Bz_ver(:,:,1),'r',(y_vec-y_offset)*1e6,Jxy_from_Bxy_ver(:,:,1),'r.-',(y_vec-y_offset)*1e6,Jxy_diff_ver(:,:,1),'r--'); title('vertical J_x (\muT)'); xlim(X_Lim); %axis([X_Lim Jx_hor_Lim]);%ylim([-100,0]);
    legend('from B_z','from B_{xy}','difference');
    subplot(2,4,6);
    plot((y_vec-y_offset)*1e6,Jxy_from_Bz_ver(:,:,2),'b',(y_vec-y_offset)*1e6,Jxy_from_Bxy_ver(:,:,2),'b.-',(y_vec-y_offset)*1e6,Jxy_diff_ver(:,:,2),'b--'); title('vertical J_y (\muT)'); xlim(X_Lim); %axis([X_Lim Jy_hor_Lim]);%ylim([2700,2800]);
    subplot(2,4,3);
    plot((x_vec-x_offset)*1e6,Jxy_from_Bz_hor(:,:,1),'r',(x_vec-x_offset)*1e6,Jxy_above_NV_hor(:,:,1),'r.-',(x_vec-x_offset)*1e6,Jxy_below_NV_hor(:,:,1),'r--'); title('horizontal J_x (\muT)'); xlim(X_Lim); %axis([X_Lim Jx_hor_Lim]);%ylim([-100,0]);
    legend('from B_z','above NV','below NV');
    subplot(2,4,4);
    plot((x_vec-x_offset)*1e6,Jxy_from_Bz_hor(:,:,2),'b',(x_vec-x_offset)*1e6,Jxy_above_NV_hor(:,:,2),'b.-',(x_vec-x_offset)*1e6,Jxy_below_NV_hor(:,:,2),'b--',(x_vec-x_offset)*1e6,PL_hor,'k:'); title('horizontal J_y (\muT)'); xlim(X_Lim); %axis([X_Lim Jy_hor_Lim]);%ylim([2700,2800]);
    subplot(2,4,7);
    plot((y_vec-y_offset)*1e6,Jxy_from_Bz_ver(:,:,1),'r',(y_vec-y_offset)*1e6,Jxy_above_NV_ver(:,:,1),'r.-',(y_vec-y_offset)*1e6,Jxy_below_NV_ver(:,:,1),'r--'); title('vertical J_x (\muT)'); xlim(X_Lim); %axis([X_Lim Jx_hor_Lim]);%ylim([-100,0]);
    legend('from B_z','above NV','below NV');
    subplot(2,4,8);
    plot((y_vec-y_offset)*1e6,Jxy_from_Bz_ver(:,:,2),'b',(y_vec-y_offset)*1e6,Jxy_above_NV_ver(:,:,2),'b.-',(y_vec-y_offset)*1e6,Jxy_below_NV_ver(:,:,2),'b--'); title('vertical J_y (\muT)'); xlim(X_Lim); %axis([X_Lim Jy_hor_Lim]);%ylim([2700,2800]);
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,10,800,400]);
    saveas(figure(81),strcat(Save_Path,'\Jxy linecuts.png'));

    figure(85);
    plot((x_vec-x_offset)*1e6,Jxy_from_Bz_hor(:,:,2),'b',(x_vec-x_offset)*1e6,Jxy_from_Bxy_hor(:,:,2),'b.-',(x_vec-x_offset)*1e6,PL_hor,'k:'); xlim(X_Lim); %axis([X_Lim Jy_hor_Lim]);%ylim([2700,2800]);
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,10,500,300]);
    xlabel('Position (\mum)'); ylabel('J_y (A/m)');
    integr_J1 = round(sum(Jxy_from_Bz_hor(:,:,2))*pixelsize_x*1e6)*1e-3;    % in mA
    integr_J2 = round(sum(Jxy_from_Bxy_hor(:,:,2))*pixelsize_x*1e6)*1e-3;   % in mA    
    legend(strcat('from B_z (integrale = ',32,num2str(integr_J1),32,'mA'),strcat('from B_{xy} (integrale = ',32,num2str(integr_J2),32,'mA'),'PL');
    saveas(figure(85),strcat(Save_Path,'\Jy linecuts.png'));

    integr = 2*sum(Bxyz_avg_hor(:,:,1)*1e-6)*pixelsize_x;
    Ampere_residue = 1-integr/(mu0*I);
    %int_Bx_theory = sum((By_conv(1,:))*1e-6)*pixelsize_x
    %int_Bz_theory = sum(abs(Bz_conv(1,:))*1e-6)*pixelsize_x
    %int_Bx = sum((Bxyz_avg_hor(1,18:75,1)-nanmean(Bxyz_avg_hor(1,90:100,1)))*1e-6)*pixelsize_x
    %int_Bz = sum(abs(Bxyz_avg_hor(:,:,3))*1e-6)*pixelsize_x*I/abs(I)
    %Ampere_residue_corrected = 1-2*int_Bx/(mu0*I)
    integr = 2*sum(Bxyz_rec_hor(:,:,1)*1e-6)*pixelsize_x;
    Ampere_residue_rec = 1-integr/(mu0*I);

    integr_ver = 2*sum(Bxyz_fit(:,:,1),2)*1e-4*pixelsize_x;
    chi_ver = 1-integr_ver/(mu0*I);

    % integr_J = sum(Jxy_tot_hor(:,:,2))*pixelsize_x*1e3
    % integr_J_above = sum(Jxy_above_hor(:,:,2))*pixelsize_x*1e3
    % integr_J_below = sum(Jxy_below_hor(:,:,2))*pixelsize_x*1e3
    % integr_J_wire = sum(Jxy_wire_hor(:,:,2))*pixelsize_x*1e3
    % integr_J_NV = sum(Jxy_NV_hor(:,:,2))*pixelsize_x*1e3
    % integr_J_outside = 1-sum(Jxy_tot_hor(:,52:64,2))/sum(Jxy_tot_hor(:,:,2))
    % J_error = 1-integr_J/I;

    integr_J = sum(Jxy_from_Bz_ver(:,:,1))*pixelsize_x*1e3;
    integr_J_wire = sum(Jxy_from_Bxy_ver(:,:,1))*pixelsize_x*1e3;
    integr_J_NV = sum(Jxy_diff_ver(:,:,1))*pixelsize_x*1e3;
    %integr_J_outside = 1-sum(Jxy_tot_ver(:,52:64,1))/sum(Jxy_tot_ver(:,:,1))
    J_error = 1-integr_J/I;

    datatosave = [(x_vec-x_offset)*1e6;Bxyz_avg_hor(1,:,1);Bxyz_avg_hor(1,:,2);Bxyz_avg_hor(1,:,3);Bxyz_rec_hor(1,:,1)]';
    save(strcat(Save_Path, '/Bxyz_hor_linecuts.txt'), 'datatosave' , '-ascii');
    datatosave = [(y_vec'-y_offset)*1e6,Bxyz_avg_ver(:,1,1),Bxyz_avg_ver(:,1,2),Bxyz_avg_ver(:,1,3)];
    save(strcat(Save_Path, '/Bxyz_ver_linecuts.txt'), 'datatosave' , '-ascii');
    datatosave = [xvec;Bx_conv;By_conv;Bz_conv]';
    save(strcat(Save_Path, '/Bxyz_theory_linecuts.txt'), 'datatosave' , '-ascii');

    datatosave = [(x_vec-x_offset)*1e6;Jxy_from_Bz_hor(1,:,1);Jxy_above_NV_hor(1,:,1);Jxy_below_NV_hor(1,:,1);Jxy_from_Bxy_hor(1,:,1);Jxy_diff_hor(1,:,1)]';
    save(strcat(Save_Path, '/Jx_hor_linecuts.txt'), 'datatosave' , '-ascii');
    datatosave = [(x_vec-x_offset)*1e6;Jxy_from_Bz_hor(1,:,2);Jxy_above_NV_hor(1,:,2);Jxy_below_NV_hor(1,:,2);Jxy_from_Bxy_hor(1,:,2);Jxy_diff_hor(1,:,2)]';
    save(strcat(Save_Path, '/Jy_hor_linecuts.txt'), 'datatosave' , '-ascii');
    datatosave = [(y_vec'-y_offset)*1e6,Jxy_from_Bz_ver(:,1,1),Jxy_above_NV_ver(:,1,1),Jxy_below_NV_ver(:,1,1),Jxy_from_Bxy_ver(:,1,1),Jxy_diff_ver(:,1,1)];
    save(strcat(Save_Path, '/Jx_ver_linecuts.txt'), 'datatosave' , '-ascii');
    datatosave = [(y_vec'-y_offset)*1e6,Jxy_from_Bz_ver(:,1,2),Jxy_above_NV_ver(:,1,2),Jxy_below_NV_ver(:,1,2),Jxy_from_Bxy_ver(:,1,2),Jxy_diff_ver(:,1,2)];
    save(strcat(Save_Path, '/Jy_ver_linecuts.txt'), 'datatosave' , '-ascii');

    integr_JfromBz_vs_y = sum(Jxy_from_Bz(:,:,2),2)*pixelsize_x*1e3;
    integr_JfromBxy_vs_y = sum(Jxy_from_Bxy(:,:,2),2)*pixelsize_x*1e3;
    integr_Jdiff_vs_y = sum(Jxy_diff(:,:,2),2)*pixelsize_x*1e3;
    PL_vs_y = sum(PL_crop(:,80:90),2); PL_vs_y = PL_vs_y/max(PL_vs_y);
    figure(82); 
    %plot((y_vec-y_offset)*1e6,integr_JfromBz_vs_y,'k',(y_vec-y_offset)*1e6,integr_JfromBxy_vs_y,'r',(y_vec-y_offset)*1e6,(y_vec-y_offset)*1e6,integr_Jdiff_vs_y,'b',-1*PL_vs_y,'k:');
    %legend('I from Bz vs y','I from Bxy vs y','I diff vs y','PL vs y');
    plot((y_vec-y_offset)*1e6,integr_JfromBxy_vs_y,'r',(y_vec-y_offset)*1e6,1*PL_vs_y,'k:');
    legend('I from Bxy vs y','PL vs y');
    xlabel('Position y (\mum)'); ylabel('Integrated current (mA)');
    datatosave = [(y_vec'-y_offset)*1e6,integr_JfromBz_vs_y,integr_JfromBxy_vs_y,integr_Jdiff_vs_y,PL_vs_y];
    save(strcat(Save_Path, '/I_vs_y.txt'), 'datatosave' , '-ascii');

    integr_JfromBz_vs_x = sum(Jxy_from_Bz(:,:,1),1)*pixelsize_y*1e3;
    integr_JfromBxy_vs_x = sum(Jxy_from_Bxy(:,:,1),1)*pixelsize_y*1e3;
    integr_Jdiff_vs_x = sum(Jxy_diff(:,:,1),1)*pixelsize_y*1e3;
    PL_vs_x = sum(PL_crop(80:90,:),1); PL_vs_x = PL_vs_x/max(PL_vs_x);
    figure(83); plot((x_vec-x_offset)*1e6,integr_JfromBz_vs_x,'k',(x_vec-x_offset)*1e6,integr_JfromBxy_vs_x,'r',(x_vec-x_offset)*1e6,integr_Jdiff_vs_x,'b',(x_vec-x_offset)*1e6,-1*PL_vs_x,'k:');
    legend('I from Bz vs x','I from Bxy vs x','I diff vs x','PL vs x');
    xlabel('Position x (\mum)'); ylabel('Integrated current (mA)');
    datatosave = [(x_vec'-x_offset)*1e6,integr_JfromBz_vs_x',integr_JfromBxy_vs_x',integr_Jdiff_vs_x',PL_vs_x'];
    save(strcat(Save_Path, '/I_vs_x.txt'), 'datatosave' , '-ascii');

    if (reconstruction_method == 2) && (Include_in_fit == 2) ||  (Include_in_fit == 3)
        Exyz_avg_hor(:,:,:) = nanmean(Exyz_sub(LinecutToPlot_horizontal-averaging_width:LinecutToPlot_horizontal+averaging_width,:,:),1);
        Exyz_avg_ver(:,:,:) = nanmean(Exyz_sub(:,LinecutToPlot_vertical-averaging_width:LinecutToPlot_vertical+averaging_width,:),2);
        Exyz_avg_hor(:,:,:) = Exyz_avg_hor(:,:,:);%-nanmean(Exyz_avg_hor(:,:,:));
        Exyz_avg_ver(:,:,:) = Exyz_avg_ver(:,:,:);%-nanmean(Exyz_avg_ver(:,:,:));

        Ex_hor_Lim = [floor(min(min(Exyz_avg_hor(:,:,1))))-1 ceil(max(max(Exyz_avg_hor(:,:,1))))+1] ;
        Ey_hor_Lim = [floor(min(min(Exyz_avg_hor(:,:,2))))-1 ceil(max(max(Exyz_avg_hor(:,:,2))))+1] ;
        Ez_hor_Lim = [floor(min(min(Exyz_avg_hor(:,:,3)))) ceil(max(max(Exyz_avg_hor(:,:,3))))] ;
        Ex_ver_Lim = [floor(min(min(Exyz_avg_ver(:,:,1))))-1 ceil(max(max(Exyz_avg_ver(:,:,1))))+1] ;
        Ey_ver_Lim = [floor(min(min(Exyz_avg_ver(:,:,2))))-1 ceil(max(max(Exyz_avg_ver(:,:,2))))+1] ;
        Ez_ver_Lim = [floor(min(min(Exyz_avg_ver(:,:,3)))) ceil(max(max(Exyz_avg_ver(:,:,3))))] ;

        if reconstruction_method == 2
            figure(82);
            subplot(2,3,1);
            plot((x_vec-x_offset)*1e6,Exyz_avg_hor(:,:,1),'r'); title('hori. E_x (V/cm)'); axis([X_Lim Ex_hor_Lim]);%ylim([-100,0]);
            subplot(2,3,2);
            plot((x_vec-x_offset)*1e6,Exyz_avg_hor(:,:,2),'b'); title('hori. E_y (V/cm)'); axis([X_Lim Ey_hor_Lim]);%ylim([2700,2800]);
            subplot(2,3,3);
            plot((x_vec-x_offset)*1e6,Exyz_avg_hor(:,:,3),'k'); title('hori. E_z (V/cm)'); axis([X_Lim Ez_hor_Lim]);%ylim([-1750,-1700]);
            subplot(2,3,4);
            plot((y_vec-y_offset)*1e6,Exyz_avg_ver(:,:,1),'r'); title('vert. E_x (V/cm)'); axis([X_Lim Ex_ver_Lim]);
            subplot(2,3,5);
            plot((y_vec-y_offset)*1e6,Exyz_avg_ver(:,:,2),'b'); title('vert. E_y (V/cm)'); axis([X_Lim Ey_ver_Lim]);
            subplot(2,3,6);
            plot((y_vec-y_offset)*1e6,Exyz_avg_ver(:,:,3),'k'); title('vert. E_z (V/cm)'); axis([X_Lim Ez_ver_Lim]);
            set(gca,'FontSize',10); set(gcf,'units','points','position',[100,10,600,400]);
            saveas(figure(82),strcat(Save_Path,'\Exyz linecuts.png'));

            datatosave = [Exyz_avg_hor(1,:,1);Exyz_avg_hor(1,:,2);Exyz_avg_hor(1,:,3)]';
            save(strcat(Save_Path, '/Exyz_hor_linecuts.txt'), 'datatosave' , '-ascii');
            datatosave = [Exyz_avg_ver(:,1,1),Exyz_avg_ver(:,1,2),Exyz_avg_ver(:,1,3)];
            save(strcat(Save_Path, '/Exyz_ver_linecuts.txt'), 'datatosave' , '-ascii');
        end
    end
end
%% Test FFT truncation artefact %%
%%%%%%%%%%%%%%%%%%%
if calc_J
    x_vec_full = -(floor(nbX/2)-0)*pixelsize_x:pixelsize_x:(ceil(nbX/2)-1)*pixelsize_x;
    y_vec_full = -(floor(nbY/2)-0)*pixelsize_y:pixelsize_y:(ceil(nbY/2)-1)*pixelsize_y;

    zp=100e-9;
    Bx_analytic0 = -mu0*I/(2*pi*width)*(atan((width-2*x_vec_full)/(2*zp))+atan((width+2*x_vec_full)/(2*zp)));
    Bz_analytic0 = mu0*I/(4*pi*width)*log(((width-2*x_vec_full).^2+(2*zp)^2)./((width+2*x_vec_full).^2+(2*zp)^2));
    Jy_analytic = I/(pi*width)*(atan((width-2*x_vec_full)/(2*1e-9))+atan((width+2*x_vec_full)/(2*1e-9)));

    % Extrapolated field
    if SizeFactorForFFT_x == 1
        Bx_analytic1 = Bx_analytic0;
        Bz_analytic1 = Bz_analytic0;
    else    
        Bx_analytic1(SizeFactorForFFT_x*nbX/2-nbX/2+1:SizeFactorForFFT_x*nbX/2+nbX/2) = Bx_analytic0;
        slope_right = Bx_analytic0(end)-Bx_analytic0(end-1);
        for ii = SizeFactorForFFT_x*nbX/2+nbX/2+1:SizeFactorForFFT_x*nbX
            Bx_analytic1(ii) = Bx_analytic1(ii-1)+slope_right;
            if Bx_analytic1(ii)*Bx_analytic1(ii-1) <= 0 || padded == 1
                Bx_analytic1(ii) = 0;   % set to 0 if change sign
            end
        end
        slope_left = Bx_analytic0(1)-Bx_analytic0(2);
        for ii = SizeFactorForFFT_x*nbX/2-nbX/2:-1:1
            Bx_analytic1(ii) = Bx_analytic1(ii+1)+slope_left;
            if Bx_analytic1(ii)*Bx_analytic1(ii+1) <= 0 || padded == 1
                Bx_analytic1(ii) = 0;   % set to 0 if change sign
            end
        end
        Bz_analytic1(SizeFactorForFFT_x*nbX/2-nbX/2+1:SizeFactorForFFT_x*nbX/2+nbX/2) = Bz_analytic0;
        slope_right = Bz_analytic0(end)-Bz_analytic0(end-1);
        for ii = SizeFactorForFFT_x*nbX/2+nbX/2+1:SizeFactorForFFT_x*nbX
            Bz_analytic1(ii) = Bz_analytic1(ii-1)+slope_right;
            if Bz_analytic1(ii)*Bz_analytic1(ii-1) <= 0 || padded == 1
                Bz_analytic1(ii) = 0;   % set to 0 if change sign
            end
        end
        slope_left = Bz_analytic0(1)-Bz_analytic0(2);
        for ii = SizeFactorForFFT_x*nbX/2-nbX/2:-1:1
            Bz_analytic1(ii) = Bz_analytic1(ii+1)+slope_left;
            if Bz_analytic1(ii)*Bz_analytic1(ii+1) <= 0 || padded == 1
                Bz_analytic1(ii) = 0;   % set to 0 if change sign
            end
        end
    end

    % FFT extrapolated field
    for ii = 1:nbY
        Bx_analytic_full(ii,:) = Bx_analytic1(1,:); 
        Bz_analytic_full(ii,:) = Bz_analytic1(1,:); 
    end
    bx_analytic = fftshift(fft2(Bx_analytic_full,SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX));  
    bz_analytic = fftshift(fft2(Bz_analytic_full,SizeFactorForFFT_y*nbY,SizeFactorForFFT_x*nbX));

    jy_analytic_fromBz = Hanning*2/mu0./exp_fac.*(1i*kx./k.*bz_analytic); 
    jy_analytic_fromBx = Hanning*2/mu0./exp_fac.*(-bx_analytic); 

    jy_analytic_fromBz(isnan(jy_analytic_fromBz)|isinf(jy_analytic_fromBz)) = 0;
    jy_analytic_fromBx(isnan(jy_analytic_fromBx)|isinf(jy_analytic_fromBx)) = 0;

    Jy_analytic_fromBz_full = real(ifft2(ifftshift(jy_analytic_fromBz))); Jy_analytic_fromBz = Jy_analytic_fromBz_full(1+crop:nbY-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
    Jy_analytic_fromBx_full = real(ifft2(ifftshift(jy_analytic_fromBx))); Jy_analytic_fromBx = Jy_analytic_fromBx_full(1+crop:nbY-crop,1+SizeFactorForFFT_x*nbX/2-nbX/2+crop:SizeFactorForFFT_x*nbX/2+nbX/2-crop); 
    Jy_analytic_fromBz = Jy_analytic_fromBz-Jy_analytic_fromBz(round(nbY/2),round(nbX/2))+Jy_analytic_fromBx(round(nbY/2),round(nbX/2)); 

    % Plot Jy
    figure(84);
    plot(x_vec*1e6,Jy_analytic_fromBz(round(nbY/2),:),'r',x_vec_full*1e6,Jy_analytic,'k:',x_vec*1e6,Jy_analytic_fromBx(round(nbY/2),:),'b');
    title('J_y (A/m)'); 
    legend('from B_z','actual','from B_x');
    set(gca,'FontSize',10); set(gcf,'units','points','position',[100,10,400,400]);
end
%% Save all parameters %%
%%%%%%%%%%%%%%%%%%%%%%%%%

param.Path = Path;
param.Path_ref = Path_ref;
param.Save_Path = Save_Path;
param.subtract_reference = subtract_reference;
param.reconstruction_method = reconstruction_method;
param.usex = useXZ;
param.Include_in_fit = Include_in_fit;
param.LinecutToPlot_horizontal = LinecutToPlot_horizontal;
param.LinecutToPlot_vertical = LinecutToPlot_vertical;
param.fit_test_horizontal_or_vertical = linecut_horizontal_or_vertical;
param.averaging_width = averaging_width;
param.Linecut_subplane = Linecut_subplane;
param.compare_OerstedField = current_on;
param.PlotODMRdata = PlotODMRdata;
param.BNV_used = BNV_used;

param.Full_ROI = Full_ROI;
param.ROI = num2str([min(ROI_square{1}),max(ROI_square{1}), min(ROI_square{2}),max(ROI_square{2})]);

%%% Device %%%
if calc_J
    param.current = current_on;
    param.width = width;
    param.zp = zp;
    param.I = I;
end

param.Bxyz_guess = num2str(Bxyz_guess);
param.uNV = reshape(uNV,[1 12]);
param.pixelsize_x = pixelsize_x;
param.pixelsize_y = pixelsize_y;

structarr = param;
%// Extract field data
fields = repmat(fieldnames(structarr), numel(structarr), 1);
values = struct2cell(structarr);

%// Convert all numerical values to strings
idx = cellfun(@isnumeric, values);
values(idx) = cellfun(@num2str, values(idx), 'UniformOutput', 0);

%// Combine field names and values in the same array
C1 = {fields{:}; values{:}};

%// Write fields to CSV file
fid = fopen(strcat(Save_Path,'/paramaters.txt'), 'wt');
fmt_str = repmat('%s:    ', 1, size(C1', 2));
fprintf(fid, [fmt_str(1:end - 1), '\n'], C1{:});
fclose(fid);

%% Plot fitted bias field
figure(86);
Bfit_bias_mag = sqrt(Bfit_bias(:,:,1).^2+Bfit_bias(:,:,2).^2+Bfit_bias(:,:,3).^2);
imagesc(Bfit_bias_mag); 
colormap(redblue()); colorbar; title('|Bias B fit| (G)'); set(gca,'xtick',[]); xlabel('x'); ylabel('y');
saveas(figure(86),strcat(Save_Path,'/Bfit_bias.png'));
save(strcat(Save_Path, '/Bfit_bias_mag.txt'), 'Bfit_bias_mag' , '-ascii');
