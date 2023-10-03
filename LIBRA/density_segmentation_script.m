function [PD, maskDense] = density_segmentation_script(img_path)

    [~, name, ~] = fileparts(img_path);
    fullName = [name '.dcm'];
    impath  = img_path;

    info    = getinfo(impath);

    % density segmentation using LIBRA
    [PD, maskDense, maskBreast] = pdensity(impath);
    
        nombreArchivoPD = 'PD.mat';
    
    % Nombre del archivo para guardar maskDense
    nombreArchivoMaskDense = 'maskDense.mat';
    
    % Guardar PD en un archivo .mat
    save(nombreArchivoPD, 'PD');
    
    % Guardar maskDense en otro archivo .mat
    save(nombreArchivoMaskDense, 'maskDense');

    PD = load(nombreArchivoPD); % Esto cargará la variable PD en el espacio de trabajo actual
    PD = PD.PD;
    % Cargar la variable maskDense desde el archivo .mat
    maskDense= load(nombreArchivoMaskDense); % Esto cargará la variable maskDense en el espacio de trabajo actual
    maskDense = maskDense.maskDense;

end
