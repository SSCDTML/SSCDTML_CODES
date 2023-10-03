function [PD_clustering] = Hybrid_segmentation_script_dcm(img_path)

    inSize = [512 512];
    
    info       = getinfo(img_path);
    im         = ffdmRead(img_path, info);

    mask = segBreast(im, info.ismlo);
    im = im.*mask;

    im = adapthisteq(im,'clipLimit',0.02,'Distribution','rayleigh');
    im = im.*mask;
    im = imresize(im, inSize);
    
    new_name = 'image_resized';
    baseFileNameIm_png = [new_name,'.png'];
    imPath_resized = fullfile(pwd, baseFileNameIm_png);

    imwrite(im2uint8(im), imPath_resized);

    % Cargar la red

    load('.\unetdepth_dsc_no_clahe_v2.mat')
    classes = {'background'; 'fgt'};

    [~,name,~] = fileparts(img_path);
    fullName = [name '.dcm'];

    I = imread(imPath_resized); %png image
    imD = ffdmRead(img_path, info); % dcm image
    imD = imresize(imD,size(I));

    maskBA = segBreast(mat2gray(imD), false);
    maskBA = imresize(maskBA,size(I));

    % FGT segmentation
    Umask   = semanticseg(I, net) == classes{2};
    UNETMask = imresize(Umask, size(I));

    K = 5;
    th = 0.5;

    maskFGT = kmeansfgt(Umask,I,K,th);
    PD_clustering = sum(maskFGT(:))/sum(maskBA(:));

    dir_seg = fullfile(pwd, "segmentacion.png");

    imwrite(im2uint8(maskFGT), dir_seg);

    PD_filename = "PD.mat";
    save(PD_filename, 'PD_clustering');

end