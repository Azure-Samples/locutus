# clean model folder
if(Test-Path ./model) { Remove-Item -Path ./model -recurse -force }
New-Item -Path ./model -ItemType Directory

if(Test-Path ./homer_model.zip) { Remove-Item ./homer_model.zip -force }

# download and expand model
$model_path = 'https://aiadvocate.blob.core.windows.net/public/homer/homer_model.zip'
Invoke-WebRequest -Uri $model_path -OutFile homer_model.zip
Expand-Archive homer_model.zip -DestinationPath ./model

