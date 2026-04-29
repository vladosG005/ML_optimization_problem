<?php
declare(strict_types=1);

$showTable = false;          // флаг для отображения таблицы
$uploadError = null;         // сообщение об ошибке загрузки

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $showTable = true;       // форма отправлена – показываем таблицу

    if (isset($_FILES['model_upload'])) {
        $model = $_FILES['model_upload'];
        if ($model['error'] === UPLOAD_ERR_OK) {
            if (!is_dir(__DIR__ . '/uploads')) {
                mkdir(__DIR__ . '/uploads', 0755, true);
            }
            $targetModel = __DIR__ . '/uploads/' . uniqid() . '.pkl';
            if (!move_uploaded_file($model['tmp_name'], $targetModel)) {
                $uploadError = 'Не удалось переместить файл с моделью.';
            }
        }
        else {
            $uploadError = 'Ошибка загрузки файла с моделью (код: ' . $model['error'] . ').';
        }
    }
    if (isset($_FILES['dataset_upload'])) {
        $dataset = $_FILES['dataset_upload'];
        if ($dataset['error'] === UPLOAD_ERR_OK) {
            if (!is_dir(__DIR__ . '/uploads')) {
                mkdir(__DIR__ . '/uploads', 0755, true);
            }
            $targetDataset = __DIR__ . '/uploads/' . uniqid() . '.csv';
            if (!move_uploaded_file($dataset['tmp_name'], $targetDataset)) {
                $uploadError = 'Не удалось переместить файл с датасетом.';
            }
        }
        else {
            $uploadError = 'Ошибка загрузки файла с датасетом (код: ' . $dataset['error'] . ').';
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
    <link rel="stylesheet" href="style.css">
</head>
<body>

<h1>Оптимизатор инференса ML-модели</h1>

<?php if ($uploadError): ?>
    <div class="error"><?= htmlspecialchars($uploadError) ?></div>
<?php endif; ?>

<form enctype="multipart/form-data" action="<?= $_SERVER['PHP_SELF'] ?>" method="post">
    <p>
        <label for="model_upload">ML-модель:</label>
        <input type="hidden" name="MAX_FILE_SIZE" value="104857600">
        <input type="file" name="model_upload" accept=".pkl"><br>
        <label for="model_upload">Датасет:</label>
        <input type="hidden" name="MAX_FILE_SIZE" value="104857600">
        <input type="file" name="dataset_upload" accept=".csv"><br>
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
                <?php
                    switch (htmlspecialchars($method)):
                        case 'Исходная модель (Python)':
                ?>
                    <td class="value"><?="{$targetModel} {$targetDataset}"?><?= shell_exec("python -W ignore ./python_benchmark/get_time.py {$targetModel} {$targetDataset}") ?></td>
                    <td class="value">2<?= shell_exec("python -W ignore ./python_benchmark/get_memory.py {$targetModel} {$targetDataset}") ?></td>
                    <td class="value">3<?= shell_exec("python -W ignore ./python_benchmark/get_accuracy.py {$targetModel} {$targetDataset}") ?></td>
                <?
                    break;
                    default:
                ?>
                    <td class="value">0</td>
                    <td class="value">0</td>
                    <td class="value">0</td>
            </tr>
                <? endswitch; ?>
            <?php endforeach; ?>
        </tbody>
    </table>
<?php endif; ?>

</body>
</html>
