function [Save_Path,PL,freq_mat,C_mat,df_mat,BNV_mat,BNV_mat_diff,I,current_on,...
    freq_mat_ref,C_mat_ref,df_mat_ref,BNV_mat_ref] = Bxyz_reconstruction_load_data_from_QDMPy(Path, Path_ref, reconstruction_method,Include_in_fit,Full_ROI,ROI,subtract_reference,Rebinning,freq_actually_measured)

Binning = str2double(Path(strfind(Path,'bin_')+4:end)); % Defines the binning from the file name for the pixel size
bibin = Binning*Rebinning;
guB = 2.8035;

% Create sub-folder
if reconstruction_method == 0
    Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_direct');
elseif reconstruction_method == 1
    Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_fit');
elseif reconstruction_method == 2
    if Include_in_fit == 0
        Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit');
    elseif Include_in_fit == 1
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_hyperfine');
    elseif Include_in_fit == 2
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_E');
     elseif Include_in_fit == 3
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_E_with_approx');
     elseif Include_in_fit == 4
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_E_NV_and_VN');
      elseif Include_in_fit == 5
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_hyperfine_and_E');
     elseif Include_in_fit == 7
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_E_and_sigma_zz');
     elseif Include_in_fit == 8
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_strain_D_shift');
     elseif Include_in_fit == 9
         Save_Path = strcat(Path,'/Bxyz_reconstruction_bin_',num2str(bibin),'_full_ODMR_fit_including_full_strain');
    end
end

if subtract_reference == 1
    Save_Path = strcat(Save_Path,'_sub_ref');
end
if exist(Save_Path,'dir') ==0
    mkdir(Save_Path)
end

% Load PL image
PL = load(strcat(Path,'/data/PL_ROI_rebinned.txt'),'-ascii');

% Load ODMR data
kk = 0;
for ii = 1:8
    if freq_actually_measured(ii) == 1
        kk = kk+1;
        freq_mat(:,:,ii) = load(strcat(Path,'/data/pos_',num2str(kk-1),'.txt'),'-ascii');
        C_mat(:,:,ii) = load(strcat(Path,'/data/amp_',num2str(kk-1),'.txt'),'-ascii');
        df_mat(:,:,ii) = load(strcat(Path,'/data/fwhm_',num2str(kk-1),'.txt'),'-ascii');
    else
        freq_mat(:,:,ii) = 0*PL;
        C_mat(:,:,ii) = 0*PL;
        df_mat(:,:,ii) = 0*PL;
    end
end

% Compute BNV
for ii = 1:4
    BNV_mat(:,:,ii) = (freq_mat(:,:,9-ii)-freq_mat(:,:,ii))/(2*guB);
    BNV_mat_diff(:,:,ii) = BNV_mat(:,:,ii);
end

% Restrict data to ROI
if Full_ROI == 1        % otherwise ROI taken as specified in settings
    ROI =  {1:size(freq_mat,1),1:size(freq_mat,2)};
end

freq_mat = freq_mat(ROI{1},ROI{2},:);
C_mat = C_mat(ROI{1},ROI{2},:);
df_mat = df_mat(ROI{1},ROI{2},:);
BNV_mat = BNV_mat(ROI{1},ROI{2},:);
BNV_mat_diff = BNV_mat_diff(ROI{1},ROI{2},:);
PL = PL(ROI{1},ROI{2},:);

% Load ref data
if subtract_reference == 1
    kk = 0;
    for ii = 1:8
        if freq_actually_measured(ii) == 1
            kk = kk+1;
            freq_mat_ref(:,:,ii) = load(strcat(Path_ref,'/data/pos_',num2str(kk-1),'.txt'),'-ascii');
            C_mat_ref(:,:,ii) = load(strcat(Path_ref,'/data/amp_',num2str(kk-1),'.txt'),'-ascii');
            df_mat_ref(:,:,ii) = load(strcat(Path_ref,'/data/fwhm_',num2str(kk-1),'.txt'),'-ascii');
        else
            freq_mat_ref(:,:,ii) = 0*PL;
            C_mat_ref(:,:,ii) = 0*PL;
            df_mat_ref(:,:,ii) = 0*PL;
        end
    end
    for ii = 1:4
        BNV_mat_ref(:,:,ii) = (freq_mat_ref(:,:,9-ii)-freq_mat_ref(:,:,ii))/(2*guB);
    end
    freq_mat_ref = freq_mat_ref(ROI{1},ROI{2},:);
    C_mat_ref = C_mat_ref(ROI{1},ROI{2},:);
    df_mat_ref = df_mat_ref(ROI{1},ROI{2},:);
    BNV_mat_ref = BNV_mat_ref(ROI{1},ROI{2},:);
    BNV_mat_diff = BNV_mat-BNV_mat_ref;
else
    freq_mat_ref = 0*freq_mat;
    C_mat_ref = 0;
    df_mat_ref = 0;
    BNV_mat_ref = 0*BNV_mat;
end

% Rebin
if Rebinning ~= 1
    freq_mat = imresize(freq_mat,1/Rebinning,'bilinear');
    C_mat = imresize(C_mat,1/Rebinning,'bilinear');
    df_mat = imresize(df_mat,1/Rebinning,'bilinear');
    BNV_mat = imresize(BNV_mat,1/Rebinning,'bilinear');
    BNV_mat_diff = imresize(BNV_mat_diff,1/Rebinning,'bilinear');
    PL = imresize(PL,1/Rebinning,'bilinear');
    freq_mat_ref = imresize(freq_mat_ref,1/Rebinning,'bilinear');
    BNV_mat_ref = imresize(BNV_mat_ref,1/Rebinning,'bilinear');
    if subtract_reference == 1
        C_mat_ref = imresize(C_mat_ref,1/Rebinning,'bilinear');
        df_mat_ref = imresize(df_mat_ref,1/Rebinning,'bilinear');
    end
end


%% Load current data %%
%%%%%%%%%%%%%%%%%%%%%%%
name_current = strcat(Path,'/IVR_data.txt');
if exist(name_current, 'file') == 2
    fileIDcurrent = fopen(name_current);
    tempcurrent = textscan(fileIDcurrent, '%f %f %f %f', 'HeaderLines',1);
    IVR_time = tempcurrent{1};
    IVR_time = IVR_time - IVR_time(1);
    IVR_volt = tempcurrent{2};
    IVR_curr = tempcurrent{3};
    IVR_res = tempcurrent{4};
    I = mean(IVR_curr)*1e-6;
    fclose(fileIDcurrent);
    current_on = 1;
else
    I = 0;
    current_on = 0;
end
