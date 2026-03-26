<?php
 declare(strict_types=1);
?>
<!DOCTYPE html>
<html lang="ru">
 <head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Оптимизатор инференса ML-модели</title>
 </head>
 <body>
     
 <h1>Оптимизатор инференса ML-модели!ZZZ</h1>

  <?php

   if (isset($_FILES['fupload'])) {
       $f = $_FILES['fupload'];
       if ($f['error'] == 0) {
           move_uploaded_file($f['tmp_name'],
           "{$f['tmp_name']}.jpg");
       }
   }

   echo '<form enctype="multipart/form-data"
         action="', $_SERVER['PHP_SELF'], '" method="post">
         <p>
         <input type="hidden" name="MAX_FILE_SIZE" value="10485760">
         <input type="file" name="fupload"><br>
         <button type="submit">Загрузить</button>
         </p>
         </form>';
  ?>
 </body>
</html>
