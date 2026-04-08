<?php
declare(strict_types=1);

$showTable = false;          // флаг для отображения таблицы
$uploadError = null;         // сообщение об ошибке загрузки

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $showTable = true;       // форма отправлена – показываем таблицу

    if (isset($_FILES['fupload'])) {
        $f = $_FILES['fupload'];
        if ($f['error'] === UPLOAD_ERR_OK) {
            if (!is_dir(__DIR__ . '/uploads')) {
                mkdir(__DIR__ . '/uploads', 0755, true);
            }
            $targetFile = __DIR__ . '/uploads/' . uniqid() . '.jpg';
            if (move_uploaded_file($f['tmp_name'], $targetFile)) {
                // Файл сохранён, инференс не производится
            } else {
                $uploadError = 'Не удалось переместить файл.';
            }
        } else {
            $uploadError = 'Ошибка загрузки файла (код: ' . $f['error'] . ').';
        }
    }
}

// Список способов переноса/оптимизации модели
$methods = [
    'Исходная модель (Python)',
    'ONNX Runtime',
    'TensorRT',
    'OpenVINO',
    'TFLite',
    'PyTorch (TorchScript)',
    'TensorFlow (SavedModel)'
];
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Оптимизатор инференса ML-модели</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 20px auto;
            padding: 0 20px;
        }
        h1 {
            color: #333;
        }
        form {
            margin: 20px 0;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        button {
            margin-top: 10px;
            padding: 6px 12px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background: #0056b3;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #007bff;
            color: white;
            text-align: center;
        }
        td:first-child {
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .empty-value {
            color: #999;
            font-style: italic;
            text-align: center;
        }
        .error {
            color: red;
            margin: 10px 0;
        }
        h2 {
            margin-top: 30px;
        }
    </style>
</head>
<body>

<h1>Оптимизатор инференса ML-модели!</h1>

<?php if ($uploadError): ?>
    <div class="error"><?= htmlspecialchars($uploadError) ?></div>
<?php endif; ?>

<form enctype="multipart/form-data" action="<?= $_SERVER['PHP_SELF'] ?>" method="post">
    <p>
        <input type="hidden" name="MAX_FILE_SIZE" value="10485760">
        <input type="file" name="fupload" accept="image/*"><br>
        <button type="submit">Загрузить</button>
    </p>
</form>

<?php if ($showTable): ?>
    <h2>Сравнение методов оптимизации инференса</h2>
    <table>
        <thead>
            <tr>
                <th>Способ переноса / Оптимизация</th>
                <th>Скорость (мс)</th>
                <th>Память (МБ)</th>
                <th>Точность (%)</th>
            </tr>
        </thead>
        <tbody>
            <?php foreach ($methods as $method): ?>
            <tr>
                <td><?= htmlspecialchars($method) ?></td>
                <td class="empty-value"></td>
                <td class="empty-value"></td>
                <td class="empty-value"></td>
            </tr>
            <?php endforeach; ?>
        </tbody>
    </table>
<?php endif; ?>

</body>
</html>
