import logging
import logging.handlers


def get_logger(name="app_logger", log_file="app.log", level=logging.DEBUG):
    """
    おすすめのロガーを作成
    - コンソールとファイルにログを出力
    - ファイルはローテーション設定（最大5MB, 最大3ファイル）
    - ログレベルを変更可能（デフォルト: DEBUG）

    :param name: ロガーの名前
    :param log_file: ログファイルのパス
    :param level: ログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :return: 設定済みのロガー
    """

    # ロガーのインスタンスを作成
    logger = logging.getLogger(name)
    logger.setLevel(level)  # ログレベル設定

    # 既存のハンドラがある場合は削除（重複防止）
    if logger.hasHandlers():
        logger.handlers.clear()

    # フォーマット定義
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # コンソールハンドラ（標準出力用）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)

    # ローテーションファイルハンドラ（5MB × 最大3ファイル）
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(log_format)

    # ロガーにハンドラを追加
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
