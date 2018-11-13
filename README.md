# music_game_result_segmentation
音ゲーのスコアリザルト画像の分類
## OverView
機械学習を用いた音ゲーのスコアリザルト画像の分類
現在は[jubeat, IIDX, SDVX]の分類のみ対応
./assets/test下の画像を分類し自動的に最適なディレクトリに振り分ける

## Requirement
- Python 3.6.6 (TensorFlowが現在3.7に対応していないため)
- <a href="https://keras.io/ja/">Keras 2.2.4</a>
- <a href="https://www.tensorflow.org/?hl=ja">TensorFlow 1.11.0</a>
- <a href="https://scikit-learn.org/stable/">scikit-learn 0.20.0</a>
- <a href="http://www.numpy.org/">numpy 1.15.3</a>

## Usage

1. 以下のコマンドを叩く(引数なしの場合は実行ファイルと同じ階層の画像を判定しに行く)

        python music_game_result_segmentation.py ['your_dir_path'] 

1. 実行ファイルと同じ階層にresultディレクトリが作成され、その下にIIDX,jubeat,SDVXディレクトリが作成、分類される。
