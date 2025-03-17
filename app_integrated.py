import streamlit as st
import cv2
import tempfile
import numpy as np
import os
import pandas as pd
import altair as alt
from pose_detection import PoseDetector
from sports_activity_recognition import SportsActivityRecognizer
from skill_level_estimation import SkillLevelEstimator

def main():
    # アプリケーションのタイトルと説明
    st.title("スポーツ骨格解析システム")
    st.markdown("""
    このアプリケーションは、画像、動画、またはリアルタイムカメラからの映像を使用して骨格を検出し、
    スポーツ活動の種類と習熟度を推定します。
    """)
    
    # サイドバーの設定
    st.sidebar.title("設定")
    app_mode = st.sidebar.selectbox(
        "モードを選択してください",
        ["ホーム", "画像解析", "動画解析", "リアルタイム解析"]
    )
    
    # 骨格検出器の初期化
    detector = PoseDetector(
        static_image_mode=True if app_mode != "リアルタイム解析" else False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # スポーツ活動認識器の初期化
    activity_recognizer = SportsActivityRecognizer(model_type="random_forest")
    
    # 習熟度推定器の初期化
    skill_estimator = SkillLevelEstimator(model_type="random_forest")
    
    # モードに応じた処理
    if app_mode == "ホーム":
        home_page()
    elif app_mode == "画像解析":
        image_analysis(detector, activity_recognizer, skill_estimator)
    elif app_mode == "動画解析":
        video_analysis(detector, activity_recognizer, skill_estimator)
    elif app_mode == "リアルタイム解析":
        realtime_analysis(detector, activity_recognizer, skill_estimator)

def home_page():
    st.markdown("""
    ## スポーツ骨格解析システムへようこそ！
    
    このアプリケーションでは、以下の機能を提供しています：
    
    1. **画像解析**: 画像をアップロードして骨格を検出
    2. **動画解析**: 動画をアップロードして骨格を検出し、スポーツ活動と習熟度を推定
    3. **リアルタイム解析**: Webカメラを使用してリアルタイムで骨格を検出し、スポーツ活動と習熟度を推定
    
    サイドバーからモードを選択して、解析を開始してください。
    
    ### 主な機能
    
    - 人体の骨格検出と可視化
    - スポーツ活動の種類の自動認識（野球、サッカー、バスケなど）
    - 習熟度の推定（初心者、中級者、上級者など）
    - 動きの質に関する詳細なメトリクス（滑らかさ、一貫性、効率性など）
    
    ### 使用技術
    
    - OpenCV: 画像・動画処理
    - MediaPipe: 骨格検出
    - TensorFlow/scikit-learn: スポーツ活動認識と習熟度推定
    - Streamlit: ユーザーインターフェース
    """)
    
    st.image("https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png", 
             caption="MediaPipe Pose Landmarksの例", width=400)

def image_analysis(detector, activity_recognizer, skill_estimator):
    st.markdown("## 画像解析")
    st.markdown("画像をアップロードして骨格を検出します。")
    
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # 画像の読み込みと骨格検出
        img = cv2.imread(tmp_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 骨格検出
        img, results = detector.find_pose(img)
        landmarks_list = detector.find_position(img, draw=True)
        
        # 特徴量の抽出
        keypoints = detector.extract_keypoints()
        
        # 結果の表示
        st.image(img, caption="骨格検出結果", use_column_width=True)
        
        # スポーツ活動認識
        st.markdown("### スポーツ活動認識")
        st.warning("注意: 画像からのスポーツ活動認識は精度が低い場合があります。より正確な結果を得るには動画またはリアルタイム解析をお試しください。")
        
        try:
            activity, confidence = activity_recognizer.predict(keypoints, is_sequence=False)
            st.write(f"予測されたスポーツ活動: **{activity}**")
            st.write(f"信頼度: {confidence:.2f}")
        except Exception as e:
            st.error(f"スポーツ活動認識中にエラーが発生しました: {str(e)}")
        
        # 習熟度推定
        st.markdown("### 習熟度推定")
        st.warning("注意: 習熟度の推定には動画またはリアルタイム映像が必要です。画像からの推定は行えません。")
        
        # 一時ファイルの削除
        os.unlink(tmp_file_path)

def video_analysis(detector, activity_recognizer, skill_estimator):
    st.markdown("## 動画解析")
    st.markdown("動画をアップロードして骨格を検出し、スポーツ活動と習熟度を推定します。")
    
    uploaded_file = st.file_uploader("動画をアップロードしてください", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # 動画の読み込みと骨格検出
        cap = cv2.VideoCapture(tmp_file_path)
        
        # 動画情報の取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 出力動画の設定
        output_path = tmp_file_path.replace('.mp4', '_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # プログレスバーの表示
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        # 動画処理
        frames_keypoints = []
        frame_idx = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # 骨格検出
            frame, results = detector.find_pose(frame)
            landmarks_list = detector.find_position(frame, draw=True)
            
            # 特徴量の抽出
            keypoints = detector.extract_keypoints()
            frames_keypoints.append(keypoints)
            
            # 動画の保存
            out.write(frame)
            
            # プログレスバーの更新
            frame_idx += 1
            progress_bar.progress(min(frame_idx / frame_count, 1.0))
        
        cap.release()
        out.release()
        
        # 処理結果の表示
        st.success("動画処理が完了しました！")
        
        # 処理済み動画の表示
        st.video(output_path)
        
        # スポーツ活動認識
        st.markdown("### スポーツ活動認識")
        
        try:
            frames_keypoints_array = np.array(frames_keypoints)
            activity, confidence = activity_recognizer.predict(frames_keypoints_array, is_sequence=True)
            st.write(f"予測されたスポーツ活動: **{activity}**")
            st.write(f"信頼度: {confidence:.2f}")
        except Exception as e:
            st.error(f"スポーツ活動認識中にエラーが発生しました: {str(e)}")
        
        # 習熟度推定
        st.markdown("### 習熟度推定")
        
        try:
            skill_level, score, quality_metrics = skill_estimator.predict(frames_keypoints_array, is_sequence=True, sport_type=activity)
            st.write(f"予測された習熟度: **{skill_level}**")
            st.write(f"習熟度スコア: {score:.2f}")
            
            # 動きの質に関するメトリクスの表示
            st.markdown("#### 動きの質に関するメトリクス")
            metrics_df = pd.DataFrame({
                'メトリクス': list(quality_metrics.keys()),
                '値': list(quality_metrics.values())
            })
            st.dataframe(metrics_df)
            
            # メトリクスの可視化
            st.markdown("#### メトリクスの可視化")
            metrics_chart = alt.Chart(metrics_df).mark_bar().encode(
                x='値',
                y=alt.Y('メトリクス', sort='-x'),
                color=alt.Color('値', scale=alt.Scale(scheme='viridis'))
            ).properties(
                width=600,
                height=300
            )
            st.altair_chart(metrics_chart)
        except Exception as e:
            st.error(f"習熟度推定中にエラーが発生しました: {str(e)}")
        
        # 一時ファイルの削除
        os.unlink(tmp_file_path)
        os.unlink(output_path)

def realtime_analysis(detector, activity_recognizer, skill_estimator):
    st.markdown("## リアルタイム解析")
    st.markdown("Webカメラを使用してリアルタイムで骨格を検出し、スポーツ活動と習熟度を推定します。")
    
    # Webカメラの起動
    run = st.checkbox("カメラを起動")
    FRAME_WINDOW = st.image([])
    
    # 結果表示用のプレースホルダー
    activity_placeholder = st.empty()
    skill_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        # キーポイントのバッファ（過去30フレーム分を保存）
        keypoints_buffer = []
        max_buffer_size = 30
        
        while run:
            success, frame = cap.read()
            if not success:
                st.error("カメラからのフレーム取得に失敗しました。")
                break
            
            # 骨格検出
            frame, results = detector.find_pose(frame)
            landmarks_list = detector.find_position(frame, draw=True)
            
            # 特徴量の抽出
            keypoints = detector.extract_keypoints()
            
            # キーポイントをバッファに追加
            keypoints_buffer.append(keypoints)
            if len(keypoints_buffer) > max_buffer_size:
                keypoints_buffer.pop(0)
            
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 結果の表示
            FRAME_WINDOW.image(frame)
            
            # バッファが十分に溜まったら解析を実行
            if len(keypoints_buffer) >= max_buffer_size:
                try:
                    # スポーツ活動認識
                    activity, confidence = activity_recognizer.predict(np.array(keypoints_buffer), is_sequence=True)
                    activity_placeholder.markdown(f"### スポーツ活動認識\n予測されたスポーツ活動: **{activity}** (信頼度: {confidence:.2f})")
                    
                    # 習熟度推定
                    skill_level, score, quality_metrics = skill_estimator.predict(np.array(keypoints_buffer), is_sequence=True, sport_type=activity)
                    skill_placeholder.markdown(f"### 習熟度推定\n予測された習熟度: **{skill_level}** (スコア: {score:.2f})")
                    
                    # 動きの質に関するメトリクスの表示
                    metrics_text = "### 動きの質に関するメトリクス\n"
                    for key, value in quality_metrics.items():
                        metrics_text += f"- {key}: {value:.2f}\n"
                    metrics_placeholder.markdown(metrics_text)
                except Exception as e:
                    activity_placeholder.error(f"解析中にエラーが発生しました: {str(e)}")
        
        cap.release()
    else:
        st.info("「カメラを起動」をチェックして、リアルタイム解析を開始してください。")

if __name__ == "__main__":
    main()
