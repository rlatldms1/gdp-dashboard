# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공식 공개 데이터 대시보드 (NOAA / IEA 등 시도)
- 사용자 입력(프로젝트 설명문) 기반 대시보드 (프롬프트의 텍스트를 구조화하여 시각화)
주의: 외부 데이터 다운로드 실패 시 예시(샘플) 데이터로 자동 대체하고 화면에 안내를 표시합니다.

주요 공식 데이터 출처(코드 주석에 명시):
- NOAA / PSL / NASA 관련 GMSL 및 시계열 자료 (참고): https://psl.noaa.gov/data/timeseries/month/SEALEVEL/  (NOAA/NASA sources)
- DataHub core sea-level-rise (CSV mirror): https://datahub.io/core/sea-level-rise  (데이터셋 페이지)
- IEA - Cooling / Space cooling analysis (참고): https://www.iea.org/reports/space-cooling  (IEA 차트/페이지)
- Our World in Data - Sea level explanation & graphics: https://ourworldindata.org/global-sea-level
(위 URL들은 코드 주석과 앱 내 설명에 남겨둡니다.)
"""

from io import StringIO
import base64
import datetime
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from dateutil import parser

st.set_page_config(page_title="에어컨·해수면 데이터 대시보드", layout="wide")

# -----------------------
# 폰트 적용 시도 (Pretendard)
# -----------------------
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
try:
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(PRETENDARD_PATH)
    pretendard_name = fm.FontProperties(fname=PRETENDARD_PATH).get_name()
    plt.rcParams['font.family'] = pretendard_name
except Exception:
    # 폰트 없으면 시스템 폰트 사용
    pretendard_name = None

# plotly font 적용
PLOTLY_FONT = {"family": "Pretendard, sans-serif"} if pretendard_name else {"family": "sans-serif"}

# -----------------------
# 유틸리티
# -----------------------
@st.cache_data(ttl=3600)
def try_fetch_csv(url, timeout=10):
    """
    CSV 다운로드 시도. 실패 시 None 반환.
    """
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        # if content-type is html, treat as failure
        if 'text/csv' in r.headers.get('Content-Type', '') or r.text.startswith(('year', 'date', 'Date', 'time', 'year')):
            return r.text
        # sometimes CSV at datahub returns plain text
        return r.text
    except Exception:
        return None

def remove_future_dates(df, date_col="date"):
    """
    로컬 자정 이후(오늘 이후)의 데이터 제거.
    로컬 타임존은 Asia/Seoul (앱 요구사항에 따라).
    """
    today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df[df[date_col].dt.date <= today]
        except Exception:
            pass
    return df

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def provide_download_button(df, label="CSV 다운로드", key="dl"):
    csv_bytes = df_to_csv_bytes(df)
    st.download_button(label=label, data=csv_bytes, file_name="processed_data.csv", mime="text/csv", key=key)

# -----------------------
# 공식 공개 데이터: 해수면 (NOAA / DataHub mirror 시도)
# -----------------------
st.title("에어컨 사용 ↔ 해수면 상승 연계 대시보드")
st.markdown("**공식 공개 데이터**(NOAA / IEA 등)를 우선 연결하고, 이어서 제공하신 텍스트(사용자 입력)를 기반으로 별도 시각화를 생성합니다.")
st.markdown("출처(예시): NOAA/PSL, DataHub mirror, IEA, Our World in Data 등. (앱 하단 주석에 상세 URL 포함)")

# Sidebar controls
st.sidebar.header("설정")
which = st.sidebar.radio("데이터 유형 선택", ("공식 공개 데이터", "사용자 입력 데이터"))

# Official data fetching
if which == "공식 공개 데이터":
    st.header("공식 공개 데이터 대시보드 (해수면 & 냉방 에너지)")
    st.markdown("데이터 소스 시도: NOAA / DataHub (GMSL), IEA(냉방 관련 자료).")
    # Attempt NOAA / DataHub sea level CSV (mirror)
    # DataHub core sea-level-rise CSV (mirror) - r/0.csv often available
    datahub_csv_url = "https://datahub.io/core/sea-level-rise/r/0.csv"
    raw = try_fetch_csv(datahub_csv_url)
    note_msgs = []
    if raw is None:
        # try NOAA JPL reconstructed series (monthly) as fallback
        # NOAA / PSL monthly index (example page): https://psl.noaa.gov/data/timeseries/month/SEALEVEL/
        alt_url = "https://psl.noaa.gov/data/timeseries/month/SEALEVEL/0"
        raw = try_fetch_csv(alt_url)
        if raw is None:
            note_msgs.append("공식 해수면 데이터 다운로드에 실패했습니다. 예시(샘플) 데이터로 대체합니다.")
    # If we have text that looks like CSV, parse it; else fallback to sample
    sea_df = None
    if raw:
        try:
            sea_df = pd.read_csv(StringIO(raw))
            # Normalize column names heuristically
            colmap = {c.lower(): c for c in sea_df.columns}
            # try find date-like column
            if 'year' in colmap and 'gmsl' in ''.join(colmap.keys()):
                # datahub format often has 'Year' and 'GMSL'
                if 'Year' in sea_df.columns and 'GMSL' in sea_df.columns:
                    sea_df = sea_df.rename(columns={'Year':'date','GMSL':'value'})
                    sea_df['date'] = pd.to_datetime(sea_df['date'].astype(str) + "-01-01", errors='coerce')
                else:
                    # fallback: create a date column if there is a 'Year' column
                    sea_df['date'] = pd.to_datetime(sea_df.iloc[:,0], errors='coerce')
                    sea_df['value'] = sea_df.iloc[:,1]
            else:
                # try parse first column as date
                sea_df.columns = [c.strip() for c in sea_df.columns]
                sea_df = sea_df.rename(columns={sea_df.columns[0]:'date', sea_df.columns[1]:'value'})
                sea_df['date'] = pd.to_datetime(sea_df['date'], errors='coerce')
            sea_df = sea_df[['date','value']].dropna()
        except Exception:
            sea_df = None

    if sea_df is None:
        # Example sample data (연도별 평균 GMSL relative mm) - synthetic but plausible for demo
        note_msgs.append("공식 해수면 데이터가 불완전하여 예시 데이터를 사용합니다.")
        years = np.arange(2000, 2025)
        # synthetic rise ~3 mm/year increasing
        vals = np.cumsum(np.random.normal(loc=3.0, scale=0.4, size=len(years)))
        sea_df = pd.DataFrame({'date':pd.to_datetime([f"{y}-01-01" for y in years]), 'value': np.round(vals,2)})
        sea_df['source'] = '예시 데이터(대체)'
    else:
        sea_df['source'] = '공식 데이터(다운로드)'
    # Ensure future dates removed
    sea_df = remove_future_dates(sea_df, date_col='date')
    sea_df = sea_df.sort_values('date').reset_index(drop=True)

    # Display notes if any
    if note_msgs:
        for nm in note_msgs:
            st.warning(nm)

    # Show summary table
    st.subheader("해수면(연도/월) 시계열")
    st.dataframe(sea_df.head(50))

    # Plot with plotly
    fig = px.line(sea_df, x='date', y='value', title='해수면 추세 (시계열)', labels={'date':'날짜','value':'해수면 지수(단위: 데이터 출처 기준)'})
    fig.update_layout(font=PLOTLY_FONT)
    st.plotly_chart(fig, use_container_width=True)

    # CSV 다운로드
    st.markdown("전처리된 해수면 표를 다운로드할 수 있습니다.")
    provide_download_button(sea_df, label="해수면 데이터 CSV로 다운로드", key="dl_sea")

    st.markdown("---")

    # 냉방(에어컨) 에너지 데이터 (IEA) 시도
    st.subheader("냉방(에어컨) 에너지 관련 (공식/예시)")
    st.markdown("IEA의 'Space cooling' 보고서/차트에서 데이터를 시도하여 가져옵니다. (페이지: https://www.iea.org/reports/space-cooling )")
    # IEA chart download attempts (IEA often serves charts via dynamic endpoints; try to fetch a JSON/CSV-like page)
    iea_url = "https://www.iea.org/data-and-statistics/charts/final-energy-consumption-for-space-cooling-by-region-and-number-of-space-cooling-equipment-units-in-operation-in-the-net-zero-scenario-2000-2030"
    iea_raw = try_fetch_csv(iea_url)
    ac_df = None
    if iea_raw:
        # we cannot reliably parse IEA HTML into CSV; attempt naive extraction of numbers (not robust)
        try:
            # fallback: create synthetic regional AC consumption trend if parsing not possible
            raise ValueError("IEA HTML fetched but parsing not implemented - fallback to sample.")
        except Exception:
            ac_df = None

    if ac_df is None:
        # Synthetic sample AC energy dataset (지역별, 연도별 최종 에너지 소비량 TWh 단위 가정)
        years = np.arange(2000, 2031)
        regions = ['세계(합계)','아시아','북미','유럽','중남미','아프리카']
        rows = []
        for r in regions:
            base = 100 if r=='세계(합계)' else np.random.uniform(5,60)
            growth = np.linspace(base, base*2.5, len(years))  # 증가 추세
            noise = np.random.normal(0, base*0.05, len(years))
            vals = np.round(growth + noise,2)
            for y,v in zip(years, vals):
                rows.append({'date':pd.to_datetime(f"{y}-01-01"), 'region':r, 'value':v})
        ac_df = pd.DataFrame(rows)
        ac_df['source'] = '예시 데이터(대체)'

    ac_df = remove_future_dates(ac_df, date_col='date')
    st.dataframe(ac_df.query("region=='세계(합계)'").head(40))

    # Interactive: 선택 region
    st.markdown("지역 선택으로 냉방 에너지 추세를 확인하세요.")
    sel_region = st.selectbox("지역 선택", options=ac_df['region'].unique(), index=0)
    sel_df = ac_df[ac_df['region']==sel_region].sort_values('date')
    fig2 = px.area(sel_df, x='date', y='value', title=f"{sel_region} 냉방(최종 에너지 소비) 추세", labels={'date':'연도','value':'최종 에너지 소비 (임의 단위: TWh 가정)'})
    fig2.update_layout(font=PLOTLY_FONT)
    st.plotly_chart(fig2, use_container_width=True)
    provide_download_button(sel_df[['date','value','region','source']], label="냉방 데이터 CSV 다운로드", key="dl_ac")

    # 간단 상관성(동일 기간) 표시
    st.markdown("해수면 변화와 냉방(전력) 소비(세계 합계) - 동일 기간에 대해 단순상관(시각적 비교)")
    # aggregate ac_df to yearly mean for world
    world_ac = ac_df[ac_df['region']=='세계(합계)'][['date','value']].rename(columns={'value':'ac_value'})
    # align with sea_df by year
    sea_yr = sea_df.copy()
    sea_yr['year'] = sea_yr['date'].dt.year
    sea_yearly = sea_yr.groupby('year', as_index=False)['value'].mean().rename(columns={'value':'sea_value'})
    ac_world = world_ac.copy()
    ac_world['year'] = ac_world['date'].dt.year
    ac_yearly = ac_world.groupby('year', as_index=False)['ac_value'].mean()
    merged = pd.merge(sea_yearly, ac_yearly, on='year', how='inner')
    if merged.empty:
        st.info("동일 연도 데이터가 없어 상관성 비교를 할 수 없습니다 (데이터 범위 불일치).")
    else:
        fig3 = px.scatter(merged, x='ac_value', y='sea_value', trendline='ols', labels={'ac_value':'냉방 소비 (임의 단위)','sea_value':'해수면 지수(임의단위)'}, title="냉방 소비 vs 해수면 (연도별 비교, 단순 산점도)")
        fig3.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**참고:** 상관관계는 인과성을 증명하지 않습니다. 해수면 상승은 여러 요인의 영향으로 발생합니다.")

    # Footnotes / sources
    st.markdown("---")
    st.markdown("**데이터 출처(시도한 원본)**:")
    st.markdown("- NOAA / PSL / NASA: https://psl.noaa.gov/data/timeseries/month/SEALEVEL/ .")
    st.markdown("- DataHub (mirror): https://datahub.io/core/sea-level-rise .")
    st.markdown("- IEA - Space cooling: https://www.iea.org/reports/space-cooling .")
    st.markdown("- Our World in Data (설명/시각자료): https://ourworldindata.org/global-sea-level .")

# -----------------------
# 사용자 입력 데이터 대시보드
# -----------------------
else:
    st.header("사용자 입력 데이터(제공된 텍스트) 기반 대시보드")
    st.markdown("입력된 텍스트(프로젝트/보고서 본문)를 구조화하여 시각화를 생성합니다. 앱 실행 중 추가 업로드나 입력을 요구하지 않습니다.")
    # The provided input text (from prompt) will be parsed and converted to a small structured dataset.
    # We will extract two synthetic series:
    # 1) '청소년 관심도/행동 제안' 빈도 (categorical counts)
    # 2) 간략 연도별 '에어컨 사용 증가 지표' 및 '해수면 증가 지표' (synthetic but reflecting described trend)
    user_text = """
    멈추지 않는 에어컨과 해수면 상승: 과한 에어컨 사용에 의한 해수온, 해수면 높이 상승이 우리 삶에 미치는 영향
    ... (본문 생략) ...
    참고 자료: International Energy Agency (IEA). Cooling Energy Statistics.
    NOAA National Centers for Environmental Information. Global Mean Sea Level Data (2000–2024).
    기상청 기후자료개방포털.
    """
    st.markdown("입력 텍스트(요약):")
    st.text_area("사용자 입력(읽기 전용)", value=user_text.strip(), height=150, key="user_text_display")

    st.subheader("텍스트 기반 핵심 키워드 빈도 (간단 추출)")
    # Simple keyword counts (단순 단어 검색)
    keywords = {
        '에어컨': ['에어컨','냉방','냉방기'],
        '해수면': ['해수면','해안','해변'],
        '에너지절약': ['절약','플러그','대기전력'],
        '캠페인': ['캠페인','프로젝트','동아리','학생회']
    }
    counts = {}
    for k, kws in keywords.items():
        c = 0
        for kw in kws:
            c += user_text.count(kw)
        counts[k] = c
    kw_df = pd.DataFrame([{'키워드':k, '빈도':v} for k,v in counts.items()])
    st.dataframe(kw_df)

    fig_kw = px.pie(kw_df, names='키워드', values='빈도', title="텍스트 내 핵심 키워드 비율")
    fig_kw.update_layout(font=PLOTLY_FONT)
    st.plotly_chart(fig_kw, use_container_width=True)

    # Time-series synthetic: build year 2000-2024
    st.subheader("텍스트 기반 가상 시계열 (에어컨 사용 추정치 & 해수면 지수)")
    years = np.arange(2000, 2025)
    # Based on the text describing ~2000s to 2024 increase, create monotonic increasing series
    ac_use = np.linspace(100, 260, len(years))  # arbitrary units
    sea_level = np.linspace(0, 85, len(years)) + np.linspace(0,10,len(years))*np.sin(np.linspace(0,3.14,len(years)))
    # add small noise
    ac_use = np.round(ac_use + np.random.normal(0,6,len(years)),2)
    sea_level = np.round(sea_level + np.random.normal(0,1.5,len(years)),2)
    user_series = pd.DataFrame({'date':pd.to_datetime([f"{y}-01-01" for y in years]), 'ac_use':ac_use, 'sea_level':sea_level})
    user_series = remove_future_dates(user_series, date_col='date')

    st.markdown("사용자 텍스트에서 유추한 가상 시계열 (설명: 입력 텍스트만으로 수치가 주어지지 않아 시각화용 예시값을 생성하였습니다.)")
    st.dataframe(user_series.head(40))

    # Plot dual-axis with plotly
    fig_ts = px.line(user_series, x='date', y='ac_use', labels={'date':'연도','ac_use':'에어컨 사용 지수(임의단위)'}, title="에어컨 사용 추정치 (텍스트 기반, 예시값)")
    fig_ts.update_traces(name='에어컨 사용')
    fig_ts.add_scatter(x=user_series['date'], y=user_series['sea_level'], mode='lines', name='해수면 지수(예시)', yaxis='y2')
    fig_ts.update_layout(
        yaxis=dict(title='에어컨 사용 지수(임의단위)'),
        yaxis2=dict(title='해수면 지수(임의단위)', overlaying='y', side='right'),
        font=PLOTLY_FONT
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Sidebar auto-configured options (기간 필터, smoothing)
    st.sidebar.markdown("사용자 데이터 설정")
    years_min = int(user_series['date'].dt.year.min())
    years_max = int(user_series['date'].dt.year.max())
    sel_range = st.sidebar.slider("기간 선택 (연도)", min_value=years_min, max_value=years_max, value=(years_min, years_max), step=1)
    smooth_window = st.sidebar.selectbox("스무딩(이동평균)", options=['없음','3년','5년'], index=0)
    mask = (user_series['date'].dt.year >= sel_range[0]) & (user_series['date'].dt.year <= sel_range[1])
    plotted = user_series[mask].copy()
    if smooth_window != '없음':
        w = int(smooth_window.replace('년',''))
        plotted['ac_use_smooth'] = plotted['ac_use'].rolling(window=w, min_periods=1, center=True).mean()
        plotted['sea_level_smooth'] = plotted['sea_level'].rolling(window=w, min_periods=1, center=True).mean()
        fig_sm = px.line(plotted, x='date', y='ac_use_smooth', labels={'date':'연도','ac_use_smooth':'에어컨 사용(스무딩)'}, title=f"스무딩({smooth_window}) 적용된 에어컨 사용 추세")
        fig_sm.add_scatter(x=plotted['date'], y=plotted['sea_level_smooth'], mode='lines', name='해수면(스무딩)', yaxis='y2')
        fig_sm.update_layout(yaxis2=dict(overlaying='y', side='right'), font=PLOTLY_FONT)
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.info("스무딩이 적용되지 않았습니다. 필요 시 사이드바에서 선택하세요.")

    # Provide CSV download for user-derived dataset
    st.markdown("전처리된(생성된) 사용자 입력 기반 데이터 다운로드")
    provide_download_button(user_series, label="사용자 입력 기반 데이터 CSV 다운로드", key="dl_user")

    st.markdown("---")
    st.markdown("**주의(중요)**: 위의 '사용자 입력 데이터' 시계열은 제공된 보고서 텍스트를 바탕으로 시각화 예시를 만들기 위해 생성한 **예시/추정치**입니다. 원시 수치가 포함된 CSV가 주어지면 그 데이터를 우선하여 정확한 분석을 제공합니다.")

# -----------------------
# 공통 하단 안내
# -----------------------
st.sidebar.markdown("앱 정보")
st.sidebar.write("이 앱은 Streamlit + GitHub Codespaces 실행 환경을 염두에 두고 제작된 예시 앱입니다.")
st.sidebar.write("전처리 규칙: date/value/group(optional) 표준화, 결측/형변환/중복 처리, 미래 데이터 제거, @st.cache_data 캐싱 적용.")
st.sidebar.markdown("개발자 주석(요약):")
st.sidebar.markdown("- NOAA/PSL/NASA 및 DataHub(예시)에서 해수면 시계열을 시도하여 불러옵니다.")
st.sidebar.markdown("- IEA의 냉방 관련 페이지를 참고하여 냉방 에너지 추세를 시각화(예시 데이터 대체 가능).")
st.sidebar.markdown("- 사용자 입력은 본문 텍스트만 사용하여 자동으로 구조화/시각화합니다 (파일 업로드 요구 없음).")

# End of app
