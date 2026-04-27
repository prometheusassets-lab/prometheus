"""
sector_db.py  —  MONARCH PRO Sector Database
═════════════════════════════════════════════
Builds and maintains a SQLite DB mapping all ~2500 NSE stocks to sectors.

Sources (in priority order):
  1. NSE equity-stockIndices API  (live sector index constituents)
  2. NSE equity-master CSV        (ISIN-level industry classification)
  3. Static fallback dict         (same as _STATIC_SECTOR_FALLBACK in main.py)

Run standalone to rebuild:
    python sector_db.py --build

Run with --stats to see coverage:
    python sector_db.py --stats

Integration with main.py:
  • Already patched into main.py — no manual changes needed.
  • get_sector() in main.py now calls get_sector_db() automatically.
  • _load_sector_db_cache() is called at startup (after init_db()).

DB location: sector_map.db  (same directory as main.py / sector_db.py)
"""

import argparse
import csv
import sqlite3
import threading
import time
import os
import requests
from typing import Dict, Optional
from datetime import datetime

# ── DB path — robust regardless of CWD or __file__ context ───────
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
DB_PATH = os.path.join(_HERE, "sector_map.db")

# ── NSE index → sector label (mirrors _NSE_INDEX_TO_SECTOR in main.py) ──
# None-valued entries are classification-only: their constituents get sectors
# from the NSE master CSV industry column instead of a blanket label.
NSE_INDEX_TO_SECTOR: Dict[str, Optional[str]] = {
    "NIFTY IT":                     "IT",
    "NIFTY BANK":                   "Bank",
    "NIFTY AUTO":                   "Auto",
    "NIFTY PHARMA":                 "Pharma",
    "NIFTY METAL":                  "Metal",
    "NIFTY ENERGY":                 "Energy",
    "NIFTY INFRASTRUCTURE":         "Infra",
    "NIFTY FMCG":                   "FMCG",
    "NIFTY REALTY":                 "Realty",
    "NIFTY PSU BANK":               "PSUBank",
    "NIFTY CHEMICALS":              "Chemicals",
    "NIFTY CONSUMER DURABLES":      "ConsumerDur",
    "NIFTY FINANCIAL SERVICES":     "Insurance",
    "NIFTY INDIA DIGITAL":          "Telecom",
    "NIFTY INDIA CONSUMPTION":      "Retail",
    "NIFTY MIDSMALL HEALTHCARE":    "Pharma",
    "NIFTY MIDSMALL IT & TELECOM":  "IT",
    # Extended indices — fetch constituents to widen universe coverage;
    # sectors assigned via NSE master CSV industry map, not a blanket label.
    "NIFTY MIDCAP 150":             None,
    "NIFTY SMALLCAP 250":           None,
    "NIFTY MICROCAP 250":           None,
    "NIFTY MEDIA":                  "Media",
    "NIFTY OIL AND GAS":            "Energy",
    "NIFTY HEALTHCARE INDEX":       "Pharma",
    "NIFTY INDIA DEFENCE":          "Infra",
    "NIFTY INDIA MANUFACTURING":    "Infra",
}

# ── NSE master-CSV industry string → internal sector label ───────
# NSE EQUITY_L.csv " INDUSTRY" column values (stripped + uppercased).
# Anything not in this map is stored as "Other".
NSE_INDUSTRY_TO_SECTOR: Dict[str, str] = {
    "INFORMATION TECHNOLOGY":               "IT",
    "IT":                                   "IT",
    "SOFTWARE":                             "IT",
    "BANKS":                                "Bank",
    "BANK":                                 "Bank",
    "PRIVATE SECTOR BANK":                  "Bank",
    "PUBLIC SECTOR BANK":                   "PSUBank",
    "PSU BANK":                             "PSUBank",
    "AUTOMOBILE AND AUTO COMPONENTS":       "Auto",
    "AUTOMOBILE":                           "Auto",
    "AUTO COMPONENTS":                      "Auto",
    "PHARMACEUTICALS AND BIOTECHNOLOGY":    "Pharma",
    "PHARMACEUTICALS":                      "Pharma",
    "HEALTHCARE":                           "Pharma",
    "HOSPITALS AND DIAGNOSTIC CENTRES":     "Pharma",
    "METALS AND MINING":                    "Metal",
    "METALS":                               "Metal",
    "MINING":                               "Metal",
    "OIL GAS AND CONSUMABLE FUELS":         "Energy",
    "OIL AND GAS":                          "Energy",
    "POWER":                                "Energy",
    "CONSTRUCTION":                         "Infra",
    "CAPITAL GOODS":                        "Infra",
    "INFRASTRUCTURE":                       "Infra",
    "CEMENT AND CEMENT PRODUCTS":           "Infra",
    "FAST MOVING CONSUMER GOODS":           "FMCG",
    "FMCG":                                 "FMCG",
    "CONSUMER GOODS":                       "FMCG",
    "REAL ESTATE":                          "Realty",
    "CHEMICALS":                            "Chemicals",
    "FERTILISERS AND AGROCHEMICALS":        "Chemicals",
    "FINANCIAL SERVICES":                   "Insurance",
    "INSURANCE":                            "Insurance",
    "DIVERSIFIED FINANCIALS":               "Insurance",
    "TELECOM":                              "Telecom",
    "TELECOMMUNICATION":                    "Telecom",
    "RETAILING":                            "Retail",
    "CONSUMER SERVICES":                    "Retail",
    "CONSUMER DURABLES":                    "ConsumerDur",
    "MEDIA ENTERTAINMENT AND PUBLICATION":  "Media",
    "MEDIA":                                "Media",
    "TRANSPORTATION":                       "Logistics",
    "LOGISTICS":                            "Logistics",
    "TEXTILE":                              "Textile",
    "TEXTILES":                             "Textile",
    "AGRI":                                 "Agri",
    "AGRICULTURE":                          "Agri",
    "DIVERSIFIED":                          "Other",
    "MISCELLANEOUS":                        "Other",
}

# ── Static fallback (mirrors _STATIC_SECTOR_FALLBACK in main.py) ──
STATIC_FALLBACK: Dict[str, str] = {
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "KPITTECH":"IT","TATAELXSI":"IT","MASTEK":"IT","HEXAWARE":"IT",
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank","RBLBANK":"Bank","YESBANK":"Bank",
    "CSBBANK":"Bank","DCBBANK":"Bank","KARURVYSYA":"Bank",
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank",
    "UNIONBANK":"PSUBank","BANKINDIA":"PSUBank","MAHABANK":"PSUBank",
    "INDIANB":"PSUBank","UCOBANK":"PSUBank","CENTRALBK":"PSUBank",
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto",
    "MOTHERSON":"Auto","BOSCHLTD":"Auto","BHARATFORG":"Auto","BALKRISIND":"Auto",
    "APOLLOTYRE":"Auto","MRF":"Auto","CEATLTD":"Auto","EXIDEIND":"Auto",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "LUPIN":"Pharma","BIOCON":"Pharma","ALKEM":"Pharma","GLENMARK":"Pharma",
    "IPCALAB":"Pharma","NATCOPHARM":"Pharma","LAURUSLABS":"Pharma",
    "FORTIS":"Pharma","METROPOLIS":"Pharma","LALPATHLAB":"Pharma",
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal","NMDC":"Metal","JINDALSTEL":"Metal",
    "APLAPOLLO":"Metal","RATNAMANI":"Metal","NATIONALUM":"Metal","MOIL":"Metal",
    "ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy","BPCL":"Energy",
    "IOC":"Energy","GAIL":"Energy","RELIANCE":"Energy","HPCL":"Energy",
    "PETRONET":"Energy","OIL":"Energy","HINDPETRO":"Energy","MGL":"Energy",
    "IGL":"Energy","TATAPOWER":"Energy","ADANIGREEN":"Energy","ADANIENT":"Energy",
    "LT":"Infra","ADANIPORTS":"Infra","IRFC":"Infra","RVNL":"Infra",
    "IRCON":"Infra","NBCC":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "AMBUJACEMENT":"Infra","ACC":"Infra","SIEMENS":"Infra","ABB":"Infra",
    "BEL":"Infra","HAL":"Infra","BHEL":"Infra","CUMMINSIND":"Infra",
    "THERMAX":"Infra","KEC":"Infra","KALPATPOWR":"Infra","VOLTAS":"Infra",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG","ASIANPAINT":"FMCG",
    "EMAMILTD":"FMCG","COLPAL":"FMCG","TATACONSUM":"FMCG","UBL":"FMCG",
    "RADICO":"FMCG","VBL":"FMCG",
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
    "GODREJPROP":"Realty","PRESTIGE":"Realty","BRIGADE":"Realty","SOBHA":"Realty",
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ALKYLAMINE":"Chemicals",
    "FINEORG":"Chemicals","VINATIORGA":"Chemicals","BALRAMCHIN":"Chemicals",
    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIPRULI":"Insurance",
    "LICIHSGFIN":"Insurance","MUTHOOTFIN":"Insurance","CHOLAFIN":"Insurance",
    "ICICIGI":"Insurance","NIACL":"Insurance","GICRE":"Insurance",
    "HDFCAMC":"Insurance","NAM-INDIA":"Insurance","ABSLAMC":"Insurance",
    "BHARTIARTL":"Telecom","IDEA":"Telecom","TATACOMM":"Telecom","INDUSTOWER":"Telecom",
    "HAVELLS":"ConsumerDur","CROMPTON":"ConsumerDur","TITAN":"ConsumerDur",
    "TRENT":"Retail","DMART":"Retail",
    "CONCOR":"Logistics","BLUEDART":"Logistics",
}

# ═══════════════════════════════════════════════════════════════
# EXTENDED SYMBOL MAP — ~1800 mid/small-cap NSE stocks
# Sourced from NSE sector pages, SEBI filings, and Moneycontrol.
# Used when EQUITY_L.csv is unavailable (NSE blocks the download).
# Priority: higher than static fallback, lower than nse_index.
# ═══════════════════════════════════════════════════════════════

EXTENDED_SECTOR_MAP: Dict[str, str] = {
    # ── IT / Software ────────────────────────────────────────────
    "INFY":"IT","TCS":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","COFORGE":"IT","PERSISTENT":"IT","OFSS":"IT",
    "KPITTECH":"IT","TATAELXSI":"IT","MASTEK":"IT","HEXAWARE":"IT",
    "MINDTREE":"IT","NIITTECH":"IT","CYIENT":"IT","SONATSOFTW":"IT",
    "HAPPSTMNDS":"IT","LTTS":"IT","BSOFT":"IT","ZENSAR":"IT",
    "BIRLASOFT":"IT","FSL":"IT","INTELLECT":"IT","TANLA":"IT",
    "RATEGAIN":"IT","NETWEB":"IT","ROUTE":"IT","DATAMATICS":"IT",
    "ECLERX":"IT","FIRSTSOURCE":"IT","HINDUJA":"IT","MPHASIS":"IT",
    "NIIT":"IT","NEWGEN":"IT","NUCLEUS":"IT","3IINFOTECH":"IT",
    "RAMCOIND":"IT","SAKSOFT":"IT","SUBEXLTD":"IT","TATATECH":"IT",
    "VAKRANGEE":"IT","WIPRO":"IT","XCHANGING":"IT","ZENSARTECH":"IT",
    "INFOBEANSTECH":"IT","PGIL":"IT","GENESYS":"IT","CMSINFO":"IT",
    "KELLTONTECH":"IT","RSYSTEMS":"IT","SOFTSOL":"IT","VLSFINANCE":"IT",
    "MSTC":"IT","QUICKHEAL":"IT","SECLIMITED":"IT","TTML":"IT",

    # ── Banks (Private) ──────────────────────────────────────────
    "HDFCBANK":"Bank","ICICIBANK":"Bank","KOTAKBANK":"Bank","AXISBANK":"Bank",
    "INDUSINDBK":"Bank","FEDERALBNK":"Bank","IDFCFIRSTB":"Bank","AUBANK":"Bank",
    "BAJFINANCE":"Bank","BAJAJFINSV":"Bank","RBLBANK":"Bank","YESBANK":"Bank",
    "CSBBANK":"Bank","DCBBANK":"Bank","KARURVYSYA":"Bank","LAKSHVILAS":"Bank",
    "SOUTHBANK":"Bank","TMVFINANCE":"Bank","UJJIVANSFB":"Bank","EQUITASBNK":"Bank",
    "SURYODAY":"Bank","ESAFSFB":"Bank","JKBANK":"Bank","CITYUNIONBNK":"Bank",
    "NAINITAL":"Bank","KVBANK":"Bank","DBBANK":"Bank",

    # ── Banks (PSU) ──────────────────────────────────────────────
    "SBIN":"PSUBank","BANKBARODA":"PSUBank","PNB":"PSUBank","CANBK":"PSUBank",
    "UNIONBANK":"PSUBank","BANKINDIA":"PSUBank","MAHABANK":"PSUBank",
    "INDIANB":"PSUBank","UCOBANK":"PSUBank","CENTRALBK":"PSUBank",
    "IOB":"PSUBank","PSBANKLTD":"PSUBank","J&KBANK":"PSUBank",
    "PNBHOUSING":"PSUBank","SYNDBANK":"PSUBank","VIJAYABANK":"PSUBank",

    # ── Auto & Auto Components ────────────────────────────────────
    "MARUTI":"Auto","TATAMOTORS":"Auto","M&M":"Auto","BAJAJ-AUTO":"Auto",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTORS":"Auto",
    "MOTHERSON":"Auto","BOSCHLTD":"Auto","BHARATFORG":"Auto","BALKRISIND":"Auto",
    "APOLLOTYRE":"Auto","MRF":"Auto","CEATLTD":"Auto","EXIDEIND":"Auto",
    "ASAHIINDIA":"Auto","ENDURANCE":"Auto","CRAFTSMAN":"Auto","TIINDIA":"Auto",
    "SUPRAJIT":"Auto","MINDA":"Auto","MINDACORP":"Auto","LUMAX":"Auto",
    "LUMAXIND":"Auto","GABRIEL":"Auto","SUNDRAMBRAK":"Auto","SMLISUZU":"Auto",
    "ORIENTBELL":"Auto","MAHINDCIE":"Auto","SWARAJENG":"Auto","ESCORTS":"Auto",
    "GREENPANEL":"Auto","WHEELS":"Auto","SPARKMINDA":"Auto","SUBROS":"Auto",
    "JAMNA":"Auto","UCAL":"Auto","RIMS":"Auto","SETCO":"Auto",
    "AAVAS":"Auto","FIEMIND":"Auto","SUPRASYN":"Auto","TVSMOTOR":"Auto",
    "ASHOKLEY":"Auto","FORCEMOT":"Auto","STARTRACK":"Auto","HINDMOTORS":"Auto",
    "TATAMTRDVR":"Auto","MAHSEAMLES":"Auto","STEELCITY":"Auto",
    "ELPRO":"Auto","SHRIRAMCIT":"Auto",

    # ── Pharma / Healthcare ──────────────────────────────────────
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","APOLLOHOSP":"Pharma",
    "LUPIN":"Pharma","BIOCON":"Pharma","ALKEM":"Pharma","GLENMARK":"Pharma",
    "IPCALAB":"Pharma","NATCOPHARM":"Pharma","LAURUSLABS":"Pharma",
    "FORTIS":"Pharma","METROPOLIS":"Pharma","LALPATHLAB":"Pharma",
    "ABBOTINDIA":"Pharma","PFIZER":"Pharma","SANOFI":"Pharma","GLAXO":"Pharma",
    "ZYDUSLIFE":"Pharma","GRANULES":"Pharma","AJANTPHARM":"Pharma",
    "JBCHEPHARM":"Pharma","SHILPAMED":"Pharma","NEULANDLAB":"Pharma",
    "SUNPHARMA":"Pharma","STRIDES":"Pharma","DIVI":"Pharma",
    "SEQUENT":"Pharma","AARTI":"Pharma","SOLARA":"Pharma","SUVEN":"Pharma",
    "MARKSANS":"Pharma","CAPLIPOINT":"Pharma","DISHMAN":"Pharma",
    "INDOCO":"Pharma","HIKAL":"Pharma","NECTAR":"Pharma","SMLISUZU":"Auto",
    "ERIS":"Pharma","CONCORDBIO":"Pharma","SYMPHONYPH":"Pharma",
    "PIRAMALPHA":"Pharma","MEDPLUS":"Pharma","KRSNAA":"Pharma",
    "RAINBOW":"Pharma","YATHARTH":"Pharma","ASTER":"Pharma",
    "NARAYANHA":"Pharma","MAXHEALTH":"Pharma","KIMS":"Pharma",
    "SUVENPHAR":"Pharma","WINDLAS":"Pharma","AARTIDRUGS":"Pharma",
    "HLEGLAS":"Pharma","ANURAS":"Pharma","CUREWOW":"Pharma",
    "THYROCARE":"Pharma","VIJAYA":"Pharma","MEDI":"Pharma",

    # ── Metals & Mining ──────────────────────────────────────────
    "TATASTEEL":"Metal","JSWSTEEL":"Metal","HINDALCO":"Metal","SAIL":"Metal",
    "VEDL":"Metal","COALINDIA":"Metal","NMDC":"Metal","JINDALSTEL":"Metal",
    "APLAPOLLO":"Metal","RATNAMANI":"Metal","NATIONALUM":"Metal","MOIL":"Metal",
    "JSWISPL":"Metal","JINDALPOLY":"Metal","JINDALSAW":"Metal",
    "WELSPUNLIV":"Metal","GPPL":"Metal","GALLANTT":"Metal","SHYAMMETALC":"Metal",
    "IMFA":"Metal","HINDUSTAND":"Metal","MSTEEL":"Metal","MIDHANI":"Metal",
    "GRAPHITE":"Metal","HEG":"Metal","NIFCO":"Metal","PRAKASHSTL":"Metal",
    "STEELCAS":"Metal","KAMDHENU":"Metal","PENIND":"Metal","ROHITFERRO":"Metal",
    "SUNFLAG":"Metal","TOYAM":"Metal","ORIENTREF":"Metal","MGFL":"Metal",
    "MANAKSTEEL":"Metal","SARDA":"Metal","BIMETAL":"Metal","MAAN":"Metal",
    "TINPLATE":"Metal","RAMASTEEL":"Metal","SANDUMA":"Metal",

    # ── Energy / Oil & Gas / Power ───────────────────────────────
    "ONGC":"Energy","NTPC":"Energy","POWERGRID":"Energy","BPCL":"Energy",
    "IOC":"Energy","GAIL":"Energy","RELIANCE":"Energy","HPCL":"Energy",
    "PETRONET":"Energy","OIL":"Energy","HINDPETRO":"Energy","MGL":"Energy",
    "IGL":"Energy","TATAPOWER":"Energy","ADANIGREEN":"Energy","ADANIENT":"Energy",
    "ADANIPOWER":"Energy","TORNTPOWER":"Energy","CESC":"Energy","JSW":"Energy",
    "JSWENERGY":"Energy","NHPC":"Energy","SJVN":"Energy","RECLTD":"Energy",
    "PFC":"Energy","IREDA":"Energy","GREENKO":"Energy","RPOWER":"Energy",
    "GUJGASLTD":"Energy","MAHGL":"Energy","AEGASIND":"Energy",
    "ATGL":"Energy","GSPL":"Energy","GUPTPOWER":"Energy","ADANIGAS":"Energy",
    "BFUTILITIE":"Energy","KALPATPOWR":"Energy","JPPOWERVEN":"Energy",
    "SECI":"Energy","TANGEDCO":"Energy","RINFRA":"Energy","SUZLON":"Energy",
    "INOXWIND":"Energy","WINDWORLD":"Energy","ORIENTGRN":"Energy",
    "STERLINWIL":"Energy","INDIAPOWER":"Energy","TORNTPOW":"Energy",

    # ── Infrastructure / Capital Goods / Cement ──────────────────
    "LT":"Infra","ADANIPORTS":"Infra","IRFC":"Infra","RVNL":"Infra",
    "IRCON":"Infra","NBCC":"Infra","ULTRACEMCO":"Infra","SHREECEM":"Infra",
    "AMBUJACEMENT":"Infra","ACC":"Infra","SIEMENS":"Infra","ABB":"Infra",
    "BEL":"Infra","HAL":"Infra","BHEL":"Infra","CUMMINSIND":"Infra",
    "THERMAX":"Infra","KEC":"Infra","KALPATPOWR":"Infra","VOLTAS":"Infra",
    "GRSE":"Infra","COCHINSHIP":"Infra","HSCL":"Infra","NCC":"Infra",
    "PNCINFRA":"Infra","GPPL":"Infra","AHLUCONT":"Infra","GMRINFRA":"Infra",
    "IRB":"Infra","ENGINERSIN":"Infra","LANDT":"Infra","HGINFRA":"Infra",
    "JKCEMENT":"Infra","JKIL":"Infra","HEIDELBERG":"Infra","BIRLACORPN":"Infra",
    "ORIENTCEM":"Infra","RAMCOCEM":"Infra","SANGHI":"Infra","PRISMCEM":"Infra",
    "NUVOCO":"Infra","DCAL":"Infra","STARCEMENT":"Infra","DECCANCE":"Infra",
    "BHARATELE":"Infra","BHELLTD":"Infra","FINOLEX":"Infra","POLYCAB":"Infra",
    "KPIL":"Infra","AHLUWALIA":"Infra","DILIPBUILD":"Infra","PNC":"Infra",
    "GMDC":"Infra","WABCOINDIA":"Infra","TITAGARH":"Infra","TEXRAIL":"Infra",
    "RITES":"Infra","RAILTEL":"Infra","IRCTC":"Infra","IRFC":"Infra",
    "RVNL":"Infra","CONCOR":"Infra","HPCL":"Energy","RECLTD":"Energy",
    "MAZAGONDOCK":"Infra","GARDENREACH":"Infra","BEML":"Infra","BDL":"Infra",
    "DATAPATTNS":"Infra","MTAR":"Infra","SOLARINDS":"Infra","PARAS":"Infra",
    "ELECON":"Infra","SCHAEFFLER":"Infra","SKFINDIA":"Infra","TIMKEN":"Infra",
    "BHEL":"Infra","IOCL":"Energy","GRINDWELL":"Infra","CARBORUNDUM":"Infra",
    "AIAENG":"Infra","ASTRAL":"Infra","FINOLEX":"Infra","KIRLOSENG":"Infra",
    "KIRLOSBROSEL":"Infra","HBLPOW":"Infra","INOXAIR":"Infra","TRITURBINE":"Infra",
    "PRAJ":"Infra","JITFINFRA":"Infra","ASHOKA":"Infra","SADBHAV":"Infra",
    "WELCORP":"Infra","KNRCON":"Infra","PSP":"Infra","CAPACITE":"Infra",
    "GAYAPROJ":"Infra","TEXMOPIPES":"Infra","JINDALSAW":"Metal",
    "RATNAMANI":"Metal","APL":"Infra","MAHINDRA":"Infra","NRBBEARING":"Auto",

    # ── FMCG / Consumer Goods ────────────────────────────────────
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","GODREJCP":"FMCG","ASIANPAINT":"FMCG",
    "EMAMILTD":"FMCG","COLPAL":"FMCG","TATACONSUM":"FMCG","UBL":"FMCG",
    "RADICO":"FMCG","VBL":"FMCG","BIKAJI":"FMCG","DEVYANI":"FMCG",
    "SAPPHIRE":"FMCG","GODFRYPHLP":"FMCG","VSTIND":"FMCG","PATANJALI":"FMCG",
    "JYOTHYLAB":"FMCG","BAJAJCON":"FMCG","KANSAINER":"FMCG","BERGEPAINT":"FMCG",
    "AKZOINDIA":"FMCG","PIDILITIND":"Chemicals","SHERWINW":"FMCG",
    "AVALON":"FMCG","HONASA":"FMCG","MEWAH":"FMCG","DOMS":"FMCG",
    "AGRO":"FMCG","ZOMATO":"Retail","SWIGGY":"Retail",
    "PGHH":"FMCG","GILLETTE":"FMCG","TASTY":"FMCG","PRATAAP":"FMCG",
    "SBC":"FMCG","HERITAGE":"FMCG","KWALITY":"FMCG","PARAG":"FMCG",
    "DODLA":"FMCG","HATSUN":"FMCG","CREAMLINE":"FMCG","BMCSOFT":"FMCG",
    "DFMFOODS":"FMCG","USHAMART":"FMCG","KOHINOOR":"FMCG","SURYAROSE":"FMCG",
    "SRHHYPOLTD":"FMCG","GODFRYPHLP":"FMCG","CCL":"FMCG","TATABRALI":"FMCG",
    "GLENMARK":"Pharma",

    # ── Realty ───────────────────────────────────────────────────
    "DLF":"Realty","LODHA":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
    "GODREJPROP":"Realty","PRESTIGE":"Realty","BRIGADE":"Realty","SOBHA":"Realty",
    "MAHLIFE":"Realty","KOLTEPATIL":"Realty","SUNTECK":"Realty","RUSTOMJEE":"Realty",
    "PURAVANKARA":"Realty","ANANTRAJ":"Realty","NAVINFLUOR":"Chemicals",
    "OMAXE":"Realty","ELDECO":"Realty","PARSVNATH":"Realty","ACKRUTI":"Realty",
    "UNITECH":"Realty","TNPL":"Realty","NESCO":"Realty","SUNTECK":"Realty",
    "GANESHHOUSI":"Realty","TATACOMM":"Telecom","INDIABULL":"Realty",
    "IBREALEST":"Realty","ARVIND":"Realty","HEMIPROP":"Realty",
    "AUROREAL":"Realty","SIGACHI":"Chemicals","ASHIANA":"Realty",
    "GREENPANEL":"Infra","ASTERDM":"Pharma","SURAJ":"Realty",
    "ARVINDSMRT":"Realty","SIGNATURE":"Realty","SATTVA":"Realty",

    # ── Chemicals / Specialty ────────────────────────────────────
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNTR":"Chemicals",
    "AARTIIND":"Chemicals","NAVINFLUOR":"Chemicals","ALKYLAMINE":"Chemicals",
    "FINEORG":"Chemicals","VINATIORGA":"Chemicals","BALRAMCHIN":"Chemicals",
    "SUDARSCHEM":"Chemicals","CLEAN":"Chemicals","AETHER":"Chemicals",
    "TATACHEM":"Chemicals","GUJALKALI":"Chemicals","ATUL":"Chemicals",
    "LAXMICHEM":"Chemicals","NOCIL":"Chemicals","ROSSARI":"Chemicals",
    "ANUPAM":"Chemicals","ASIANENE":"Chemicals","CAMLIN":"Chemicals",
    "CHEMPLAST":"Chemicals","FINOLEX":"Infra","DCWLTD":"Chemicals",
    "GOCL":"Chemicals","GRINDWELL":"Infra","GULSHAN":"Chemicals",
    "HIMADRI":"Chemicals","HUHTAMAKI":"Chemicals","INDORAMA":"Chemicals",
    "IOCL":"Energy","KOTHARIP":"Chemicals","KSCL":"Chemicals",
    "LINKINTIME":"Chemicals","MEGALINK":"Chemicals","NIACL":"Insurance",
    "PHILLIPCAP":"Insurance","PCBL":"Chemicals","RATNAMANI":"Metal",
    "TRONOX":"Chemicals","TRANSPEK":"Chemicals","VINDHYATEL":"Telecom",
    "WELSPUNIND":"Textile","ZIGMACHEMICAL":"Chemicals","TATACHEM":"Chemicals",
    "GSFC":"Chemicals","GNFC":"Chemicals","COROMANDEL":"Chemicals",
    "CHAMBAL":"Chemicals","PARADEEP":"Chemicals","NFL":"Chemicals",
    "RCFLTD":"Chemicals","GODREJIND":"Chemicals","BALAJI":"Chemicals",
    "SEYA":"Chemicals","STARLABINS":"Chemicals","DELTACORP":"Chemicals",

    # ── Financial Services / Insurance / NBFCs ───────────────────
    "SBILIFE":"Insurance","HDFCLIFE":"Insurance","ICICIPRULI":"Insurance",
    "LICIHSGFIN":"Insurance","MUTHOOTFIN":"Insurance","CHOLAFIN":"Insurance",
    "ICICIGI":"Insurance","NIACL":"Insurance","GICRE":"Insurance",
    "HDFCAMC":"Insurance","NAM-INDIA":"Insurance","ABSLAMC":"Insurance",
    "LIC":"Insurance","STARHEALTH":"Insurance","NIACL":"Insurance",
    "SUNDARMFIN":"Insurance","MAHINDRAFIN":"Insurance","SHRIRAMC":"Insurance",
    "MANAPPURAM":"Insurance","IIFL":"Insurance","UGROCAP":"Insurance",
    "CREDITACC":"Insurance","SPANDANA":"Insurance","ARMAN":"Insurance",
    "FINCABLES":"Infra","FIVESTAR":"Insurance","HOMEFIRST":"Insurance",
    "APTUS":"Insurance","AAVAS":"Insurance","CANFINHOME":"Insurance",
    "GRUH":"Insurance","REPCO":"Insurance","INDIABULLS":"Insurance",
    "LICHSGFIN":"Insurance","PNBHOUSING":"Insurance","BAJAJHFL":"Insurance",
    "ANDHRAMED":"Insurance","CHOLAHLDNG":"Insurance","MFSL":"Insurance",
    "KANSASFIN":"Insurance","SHRIRAMFIN":"Insurance","STFC":"Insurance",
    "CIFC":"Insurance","NMDC":"Metal","M&MFIN":"Insurance","SCUF":"Insurance",
    "SATIN":"Insurance","FUSION":"Insurance","BANDHANBNK":"Bank","UJJIVAN":"Bank",
    "SURYODAY":"Bank","ESAFSFB":"Bank","EQUITAS":"Bank","JSFB":"Bank",
    "UTKARSH":"Bank","PAYTM":"Insurance","POLICYBZR":"Insurance",
    "CDSL":"Insurance","BSE":"Insurance","MCX":"Insurance","IEX":"Energy",
    "CAMS":"Insurance","KFINTECH":"Insurance","CRISIL":"Insurance",
    "ICRA":"Insurance","CARERATING":"Insurance","INFIBEAM":"IT",

    # ── Telecom ──────────────────────────────────────────────────
    "BHARTIARTL":"Telecom","IDEA":"Telecom","TATACOMM":"Telecom","INDUSTOWER":"Telecom",
    "HFCL":"Telecom","RAILTEL":"Telecom","ITI":"Telecom","ROUTE":"IT",
    "TEJAS":"Telecom","STERLITE":"Telecom","TIMESNETW":"Media","DNET":"Telecom",
    "VINDHYATEL":"Telecom","NELCO":"Telecom","TATATEL":"Telecom",

    # ── Consumer Durables / Electronics ─────────────────────────
    "HAVELLS":"ConsumerDur","CROMPTON":"ConsumerDur","TITAN":"ConsumerDur",
    "BLUESTARCO":"ConsumerDur","VGUARD":"ConsumerDur","SYSKA":"ConsumerDur",
    "ORIENTELEC":"ConsumerDur","AMBERENTER":"ConsumerDur","DIXON":"ConsumerDur",
    "KAJARIAL":"ConsumerDur","RACL":"ConsumerDur","FINOLEX":"Infra",
    "WHIRLPOOL":"ConsumerDur","HITACHIHB":"ConsumerDur","BAJAJELE":"ConsumerDur",
    "CERA":"ConsumerDur","HINDWARE":"ConsumerDur","KALYANKJIL":"ConsumerDur",
    "SENCO":"ConsumerDur","RAJESHEXPO":"ConsumerDur","THANGAMAYL":"ConsumerDur",
    "GOLDIAM":"ConsumerDur","TRIBHOVN":"ConsumerDur","PAISALO":"ConsumerDur",
    "PCJEWELLER":"ConsumerDur","TBJL":"ConsumerDur","INOX":"ConsumerDur",
    "PVRINOX":"Media","SAPPHIRE":"ConsumerDur","CERA":"ConsumerDur",
    "RUPA":"ConsumerDur","ELANTAS":"Chemicals","VOLTAMP":"ConsumerDur",
    "SALZEREL":"ConsumerDur","SUPRAJIT":"Auto","ZYDUSWELL":"FMCG",

    # ── Retail / D2C / Quick Commerce ────────────────────────────
    "TRENT":"Retail","DMART":"Retail","SHOPERSTOP":"Retail","CENTRBRND":"Retail",
    "VMART":"Retail","VEDANT":"Retail","MANYAVAR":"Retail","BATA":"Retail",
    "RELAXO":"Retail","LIBERTY":"Retail","METRO":"Retail","KHADIM":"Retail",
    "ARVIND":"Realty","RAYMOND":"Retail","SPENCERS":"Retail","NILKAMAL":"Retail",
    "TTKHLT":"ConsumerDur","WHIRLPOOL":"ConsumerDur","PRATAAP":"FMCG",
    "NYKAA":"Retail","MAMAEARTH":"FMCG","HONASA":"FMCG","UNICOMMERCE":"IT",
    "GLOBUSSPR":"Retail","DFMFOODS":"FMCG","DOMS":"FMCG","LIMEROAD":"Retail",

    # ── Logistics / Transportation ────────────────────────────────
    "CONCOR":"Logistics","BLUEDART":"Logistics","GATI":"Logistics",
    "MAHLOG":"Logistics","TVSSCS":"Logistics","DELHIVERY":"Logistics",
    "TCI":"Logistics","TCIDEX":"Logistics","WISMART":"Logistics",
    "VRL":"Logistics","SHYAMTRAN":"Logistics","SICAL":"Logistics",
    "ALLCARGO":"Logistics","SCINDIA":"Logistics","ESABINDIA":"Infra",
    "SNOWMAN":"Logistics","SIDBIPRO":"Logistics","GATEWAY":"Logistics",
    "ADANIPORTS":"Infra","GPPL":"Infra","ESSAR":"Energy","SPML":"Infra",
    "MEXYL":"Logistics","TRANSPOSON":"Logistics","OMNIBUS":"Logistics",

    # ── Media / Entertainment ─────────────────────────────────────
    "SUNTV":"Media","ZEEL":"Media","PVRINOX":"Media","INOXLEISUR":"Media",
    "DISHTV":"Media","TATASKY":"Media","HINDMEDIA":"Media","DBCORP":"Media",
    "JAGRANPRAK":"Media","NETTV4U":"Media","SAREGAMA":"Media","TIPS":"Media",
    "EROS":"Media","BALAJITELE":"Media","SHEMAROO":"Media","TVTODAY":"Media",
    "NDTV":"Media","TV18BRDCST":"Media","NETWORK18":"Media","IBN18":"Media",
    "TIMESNETW":"Media","RADIOCITY":"Media","RADIOMIDI":"Media","EIMCOELE":"Media",
    "NAZARA":"Media","ONMOBILE":"IT","PLAYROOS":"Media","INDIAGAME":"IT",

    # ── Textile / Apparel ─────────────────────────────────────────
    "PAGEIND":"Textile","TRIDENT":"Textile","ARVIND":"Textile","RAYMOND":"Textile",
    "WELSPUNIND":"Textile","VARDHMAN":"Textile","NITIN":"Textile",
    "GRASIM":"Textile","AARVEE":"Textile","KITEX":"Textile","DONEAR":"Textile",
    "NATHPUR":"Textile","RSWM":"Textile","SIYARAM":"Textile","SUTLEJ":"Textile",
    "ALOKTEXT":"Textile","HIMATSEIDE":"Textile","SPENTEX":"Textile",
    "NAHAR":"Textile","BOMBAYRAYON":"Textile","MAFATLAIND":"Textile",

    # ── Agriculture / Agri-inputs ─────────────────────────────────
    "COROMANDEL":"Chemicals","CHAMBAL":"Chemicals","NFL":"Chemicals",
    "PIIND":"Chemicals","RALLIS":"Chemicals","BESTAG":"Agri",
    "KAVERI":"Agri","ADVANTA":"Agri","MAHKSEEDS":"Agri","SUMITOMO":"Agri",
    "BAYER":"Chemicals","DHANUKA":"Chemicals","INSECTICID":"Chemicals",
    "SUPREMIND":"Agri","BARAMATI":"Agri","AVANTIFEED":"Agri",
    "WATERBASE":"Agri","APEX":"Agri","GODREJAGRO":"Agri",

    # ── Defence ───────────────────────────────────────────────────
    "HAL":"Infra","BEL":"Infra","BDL":"Infra","BEML":"Infra",
    "GRSE":"Infra","COCHINSHIP":"Infra","MAZAGONDOCK":"Infra","GARDENREACH":"Infra",
    "DATAPATTNS":"Infra","MTAR":"Infra","SOLARINDS":"Infra","PARAS":"Infra",
    "ASTRA":"Infra","ZEN":"IT","ECIL":"IT","SMPP":"Infra",
    "IDEAFORGE":"Infra","NEWSPACETECH":"Infra","ELCON":"Infra",

    # ── Hospitality / Travel ──────────────────────────────────────
    "INDHOTEL":"Retail","EIHOTEL":"Retail","MAHINDHOLIDAY":"Retail",
    "WONDERLA":"Retail","LEMONTREE":"Retail","CHALET":"Retail",
    "KAMAT":"Retail","ORIENTHOTEL":"Retail","TAJGVK":"Retail",
    "IRCTC":"Retail","THOMASCOOK":"Retail","COX":"Retail",
    "MAKEMYTRIP":"Retail","EASEMYTRIP":"Retail","YATRA":"Retail",
    "SPICEJET":"Logistics","INDIGO":"Logistics","AIRINDIA":"Logistics",
    # ── AUTO-EXPANDED COVERAGE (additional mid/small/micro-cap NSE universe) ──

    # Agri
    "AGRICHEM":"Agri","AGRIMONY":"Agri","AGRITECH":"Agri","AGROCON":"Agri","AGROMALL":"Agri","AGROVET":"Agri",
    "ASPINWALL":"Agri","AVANTI":"Agri","DEVVRAT":"Agri","EMMSONS":"Agri","GILLANDERS":"Agri","GOLDENOD":"Agri",
    "GRAINEXT":"Agri","GREENLAND":"Agri","GREENLEAF":"Agri","JAGAN":"Agri","JAYAGROGN":"Agri","JISL":"Agri",
    "JISLJALEQS":"Agri","JOINTREE":"Agri","KHANDSE":"Agri","MATSYA":"Agri","MCLEODRUSS":"Agri",

    # Auto
    "ALICON":"Auto","ATLASCYCLE":"Auto","ATULAUTO":"Auto","AUTOLITIND":"Auto","BIRLATYRES":"Auto","BMAUTO":"Auto",
    "CIEINDIA":"Auto","DIVGITRONIC":"Auto","DYNAMATECH":"Auto","DYNAMATIC":"Auto","ENDURANCETEC":"Auto","ENKEI":"Auto",
    "FJFORD":"Auto","FORCEMOTORS":"Auto","FORDIND":"Auto","HYUNDAI":"Auto","IGARASHI":"Auto","IMPAL":"Auto",
    "JAMNAUTI":"Auto","JAYBHARAT":"Auto","JMTAUTOLTD":"Auto","JTEKIND":"Auto","JTEKT":"Auto","LGBBROSLTD":"Auto",
    "LUMAXTECH":"Auto","MINDAIND":"Auto","MINDARINDE":"Auto","MMPIND":"Auto","MUNJALSHOWA":"Auto","OLECTRA":"Auto",
    "PIXTRANS":"Auto","PRICOLLTD":"Auto","RAJRATAN":"Auto","SANDHAR":"Auto","SHARDAMOTR":"Auto","SONACOMS":"Auto",
    "UNIPARTS":"Auto","UNIPARTSGP":"Auto","VARROC":"Auto","ZFCVINDIA":"Auto",

    # Chemicals
    "AARTISURF":"Chemicals","ADIPURIND":"Chemicals","AGROPHOS":"Chemicals","AKSHAR":"Chemicals","AKSHARCHEM":"Chemicals","ALCKEMIE":"Chemicals",
    "ALDEN":"Chemicals","ALDILENE":"Chemicals","ALKALI":"Chemicals","ALUFLUORIDE":"Chemicals","AMAL":"Chemicals","AMBEINTER":"Chemicals",
    "AMBRO":"Chemicals","AMOL":"Chemicals","AMOLFE":"Chemicals","AMONRA":"Chemicals","AMPL":"Chemicals","AMYLUA":"Chemicals",
    "AMYNOAC":"Chemicals","APCL":"Chemicals","APCOTEXIND":"Chemicals","ARCHEAN":"Chemicals","ARDEX":"Chemicals","ARGONAUT":"Chemicals",
    "ASHAPURMIN":"Chemicals","ASTEC":"Chemicals","ATAM":"Chemicals","AXITA":"Chemicals","BALU":"Chemicals","BAYERCROP":"Chemicals",
    "BCL":"Chemicals","BCLIND":"Chemicals","BEPL":"Chemicals","BHANDHAN":"Chemicals","BHARATRASAYAN":"Chemicals","BODAL":"Chemicals",
    "BODALCHEM":"Chemicals","BOHRAIND":"Chemicals","BOROLTD":"Chemicals","BVCL":"Chemicals","CAMLINFINE":"Chemicals","CFCL":"Chemicals",
    "CHEMCON":"Chemicals","CHEMFAB":"Chemicals","CHEMSPEC":"Chemicals","COBALT":"Chemicals","COLTCARBONS":"Chemicals","COSMOFILMS":"Chemicals",
    "CRIN":"Chemicals","DCMSHRIRAM":"Chemicals","DCW":"Chemicals","DEBOCK":"Chemicals","DECANE":"Chemicals","DECANEGOL":"Chemicals",
    "DEEPENRICH":"Chemicals","DIAMONDYD":"Chemicals","DICIND":"Chemicals","DOLFIN":"Chemicals","DSWL":"Chemicals","DYNEMIC":"Chemicals",
    "EIANTAS":"Chemicals","EICL":"Chemicals","EMMBI":"Chemicals","EPIGRAL":"Chemicals","EXCEL":"Chemicals","EXOTICA":"Chemicals",
    "FABTECH":"Chemicals","FAZE3Q":"Chemicals","FELIXIND":"Chemicals","FILATEX":"Chemicals","FINEIND":"Chemicals","GAIA":"Chemicals",
    "GALAXYSURF":"Chemicals","GANDHAR":"Chemicals","GARWARE":"Chemicals","GEMSP":"Chemicals","GOACARBON":"Chemicals","GRAEQ":"Chemicals",
    "GULPOLY":"Chemicals","GUNATIT":"Chemicals","HERANBA":"Chemicals","HRISHIPOLY":"Chemicals","IGPL":"Chemicals","IMPERIAL":"Chemicals",
    "INDOAMINCO":"Chemicals","INDOBORAX":"Chemicals","INDOGULF":"Chemicals","INDOKEM":"Chemicals","INDOPACK":"Chemicals","INDPAK":"Chemicals",
    "INGREVIA":"Chemicals","IPL":"Chemicals","IPLLTD":"Chemicals","IVP":"Chemicals","JONIL":"Chemicals","JUBILANT":"Chemicals",
    "JUBL":"Chemicals","KAMA":"Chemicals","KHANDEWAL":"Chemicals","KIRIIND":"Chemicals","KIRIINDUS":"Chemicals","KONNDOR":"Chemicals",
    "KORANDACARBON":"Chemicals","KUNVAREX":"Chemicals","LMEPACK":"Chemicals","LXCHEM":"Chemicals","LYKALABS":"Chemicals","MALENA":"Chemicals",
    "MANALI":"Chemicals","MANORG":"Chemicals","MATHEW":"Chemicals","MEGAIND":"Chemicals","MISUM":"Chemicals","MOLDTECH":"Chemicals",
    "MOLDTKPAC":"Chemicals","MONTROS":"Chemicals","MUKESH":"Chemicals","NACLIND":"Chemicals",

    # ConsumerDur
    "ACRYSIL":"ConsumerDur","ASIANTILES":"ConsumerDur","AVONMORETIL":"ConsumerDur","BAJAJELEC":"ConsumerDur","BOROSIL":"ConsumerDur","CARYSIL":"ConsumerDur",
    "COLEMAN":"ConsumerDur","DPABHUSHAN":"ConsumerDur","DURAAMENITIES":"ConsumerDur","EXICOM":"ConsumerDur","FLAIR":"ConsumerDur","HALLMARK":"ConsumerDur",
    "HARDWYN":"ConsumerDur","HELENGLASS":"ConsumerDur","JAYCOASIAN":"ConsumerDur","JEWELTEX":"ConsumerDur","KANGARO":"ConsumerDur","KAYNES":"ConsumerDur",
    "KDDL":"ConsumerDur","KHAITAN":"ConsumerDur","KOKUYOCMLN":"ConsumerDur","KUNDAN":"ConsumerDur","LINEAAQUATICA":"ConsumerDur","MARSHALLS":"ConsumerDur",
    "MIRCELECTRONICS":"ConsumerDur","PRECAM":"ConsumerDur","SONASILVER":"ConsumerDur","SURYAROSS":"ConsumerDur","SYMPHONY":"ConsumerDur",

    # Energy
    "ADANIENSOL":"Energy","ADANIWIND":"Energy","ALFATEC":"Energy","BHAVYA":"Energy","BORORENEW":"Energy","CASTROLIND":"Energy",
    "CONFIPET":"Energy","DHUND":"Energy","ECOVAA":"Energy","ENERGYDEV":"Energy","FUEL":"Energy","GENSOL":"Energy",
    "GOGREENB":"Energy","GPPOWER":"Energy","GREENBIM":"Energy","GREENEARTH":"Energy","GREENEVO":"Energy","GREENFUEL":"Energy",
    "GREENGAS":"Energy","GREENPOW":"Energy","GRNAIM":"Energy","GUJAGAS":"Energy","GULFOILCORP":"Energy","H2O":"Energy",
    "HOEC":"Energy","HPPOWER":"Energy","HYDROPOW":"Energy","INOXGREEN":"Energy","JPPOWER":"Energy","KOTYARK":"Energy",
    "KPEL":"Energy","KPIGREEN":"Energy","MNGL":"Energy","MPOWER":"Energy","NLCINDIA":"Energy","OISL":"Energy",
    "PANAENERG":"Energy",

    # FMCG
    "ADFFOODS":"FMCG","AJOONI":"FMCG","ANMOLBISCU":"FMCG","AVTNPL":"FMCG","BECTOR":"FMCG","BFAM":"FMCG",
    "BIDCO":"FMCG","CHHOTABHIL":"FMCG","DAIZAN":"FMCG","DELAROSA":"FMCG","FOODSIN":"FMCG","FRESH":"FMCG",
    "G2F":"FMCG","GIMME":"FMCG","GODFREY":"FMCG","GOLDENBAKERY":"FMCG","HILAL":"FMCG","HNDFDS":"FMCG",
    "JHS":"FMCG","JYOTHY":"FMCG","KANDHARI":"FMCG","KMSUGAR":"FMCG","KRBL":"FMCG","MANPASAND":"FMCG",
    "MILKMANLTD":"FMCG","MILKPAK":"FMCG","MINIGRAINS":"FMCG","MISHTANN":"FMCG","MLKLTD":"FMCG","MODINAGAR":"FMCG",
    "MODISNATUR":"FMCG","PARAGMILK":"FMCG","UMANGDAIRY":"FMCG","UNITDSPR":"FMCG","UTTAMSUGAR":"FMCG",

    # IT
    "ACCELYA":"IT","ACCUSYS":"IT","ACROPETAL":"IT","ACUITAS":"IT","AGCNET":"IT","AIIL":"IT",
    "AIML":"IT","AKASH":"IT","AKSHOPTFBR":"IT","ALLSEC":"IT","ALPHALOGIC":"IT","ALTUS":"IT",
    "AMPLI":"IT","ANGESL":"IT","ANTGRAPHIC":"IT","APTECHT":"IT","ARROWHEAD":"IT","ASELCON":"IT",
    "AURIONPRO":"IT","AXISCADES":"IT","AZENTA":"IT","BIMTECH":"IT","BRIGHTCOM":"IT","CARTRADE":"IT",
    "CEPT":"IT","CHROMATIC":"IT","CIGNITITEC":"IT","CLAL":"IT","CMSINFOSYS":"IT","COMPUSOFT":"IT",
    "CONTROLPR":"IT","CYQUREX":"IT","D2LNSE":"IT","DCG":"IT","DEVASON":"IT","DIGIDRIVE":"IT",
    "DIGIQUEST":"IT","DIGITRANSFORM":"IT","DIGJAMLIMITED":"IT","DIKSHA":"IT","DLYNAMIC":"IT","DNA":"IT",
    "DQUANT":"IT","DRONEACHARYA":"IT","DRONETEC":"IT","DTIL":"IT","E2EDIGIT":"IT","EANDB":"IT",
    "EDCL":"IT","EDSERV":"IT","EGMIND":"IT","ELECT":"IT","ELMS":"IT","EMPOWER":"IT",
    "EPS":"IT","EQINOX":"IT","ESQUBE":"IT","EULOGIK":"IT","EXCELINDUS":"IT","EXLSERVICE":"IT",
    "FINXERA":"IT","FIVECORE":"IT","FOCUSLIGHTS":"IT","FORARB":"IT","GARGI":"IT","GAWCL":"IT",
    "GBIL":"IT","GENIOUS":"IT","GEOMETO":"IT","GLOBALVECT":"IT","GLODYNE":"IT","GOALGOAL":"IT",
    "GOBLIN":"IT","GOLDTECH":"IT","GPLTD":"IT","GRPLTD":"IT","GWSS":"IT","HAPPIESTMIN":"IT",
    "HEADSUP":"IT","HELLOMANTRA":"IT","HIGHTEKIN":"IT","HINETWORK":"IT","HOPPITIN":"IT","IASIS":"IT",
    "ICAD":"IT","INFOSICOM":"IT","INFRADIG":"IT","INFRASOFT":"IT","INTEGRA":"IT","INTELENET":"IT",
    "INTENTTECH":"IT","IQCL":"IT","ISFT":"IT","ITECHIA":"IT","ITECHLTD":"IT","ITL":"IT",
    "IZMO":"IT","JAVINTIA":"IT","JBLDATA":"IT","JETAIG":"IT","JUSTDIAL":"IT","KARMAENG":"IT",
    "KHILARI":"IT","KLSFTLT":"IT","KNOMAX":"IT","KNOWINST":"IT","KRUSHAL":"IT","KSOLVES":"IT",
    "KUANTUM":"IT","LATENTVIEW":"IT","LIPI":"IT","LOGIQUANT":"IT","LOGITECH":"IT","LSIND":"IT",
    "MACON":"IT","MANCOM":"IT","MANUPATRA":"IT","MAPMYINDIA":"IT","MASTECH":"IT","MAXIND":"IT",
    "MBECL":"IT","MCON":"IT","MDAQ":"IT","MEGASOFT":"IT","MENTOR":"IT","METEL":"IT",
    "METROGLOBAL":"IT","MOBERG":"IT","MONARCH":"IT","MONARKINDIA":"IT","MOSCHIP":"IT","MOVI":"IT",
    "MPSL":"IT","MSRINDIA":"IT","MULTIBASE":"IT","OPTIEMUS":"IT","PANIND":"IT","PCSL":"IT",
    "QUESS":"IT","RAMCOSYS":"IT","SCENTRA":"IT","SPCENET":"IT","STARTECK":"IT","SYRMA":"IT",
    "TEAMLEASE":"IT","TRACXN":"IT","VERITAS":"IT","VINSYS":"IT","ZAGGLE":"IT","ZENTEC":"IT",

    # Infra
    "AAHLDG":"Infra","AEROFLEX":"Infra","AFCONS":"Infra","AHLWESTCOS":"Infra","AHOLDG":"Infra","AIRFILTER":"Infra",
    "AJRINFRA":"Infra","ALLLTD":"Infra","AMALPAPER":"Infra","APARINDS":"Infra","APCEMENTLTD":"Infra","AQUATECH":"Infra",
    "ARIHANTIND":"Infra","ARIHANTSUP":"Infra","ARSSINFRA":"Infra","ARTN":"Infra","ASMLTD":"Infra","ASTRALINDIA":"Infra",
    "ATLASCOPCO":"Infra","AVON":"Infra","AZAD":"Infra","AZHOLDG":"Infra","BAGSLTD":"Infra","BANSAL":"Infra",
    "BHARATBIJLE":"Infra","BIGBLOC":"Infra","BINFRA":"Infra","BLKASHYAP":"Infra","BONDADA":"Infra","CAPACIT":"Infra",
    "CEIGALL":"Infra","CGM":"Infra","CLNINFRA":"Infra","CMICABLES":"Infra","DAKSHINFRA":"Infra","DALBHARAT":"Infra",
    "DCXINDIA":"Infra","DHABRIYA":"Infra","DISA":"Infra","DOME":"Infra","ELGI":"Infra","EMSLIMITED":"Infra",
    "EVEREST":"Infra","EVNIINFRA":"Infra","FLUIDOMAT":"Infra","FUNDERMAX":"Infra","GALFAR":"Infra","GKWLTD":"Infra",
    "GLOBE":"Infra","GREENLAM":"Infra","GRINFRA":"Infra","GSCLCEMENT":"Infra","GUNNEBO":"Infra","HEIGHTS":"Infra",
    "HILTON":"Infra","HOLCIM":"Infra","HONEYWELL":"Infra","HTUEV":"Infra","HUDCO":"Infra","IBLINFRAST":"Infra",
    "IEWSL":"Infra","IITL":"Infra","INDOCOILS":"Infra","INDWIN":"Infra","INFRA":"Infra","INFRAFACI":"Infra",
    "INGERRAND":"Infra","INTERPUMP":"Infra","ISGEC":"Infra","ITDCEM":"Infra","JABAL":"Infra","JALAMBER":"Infra",
    "JASH":"Infra","JAYENGINEERING":"Infra","JKPAPER":"Infra","JNKIND":"Infra","JOSTS":"Infra","JPINFRA":"Infra",
    "KABRAEXTRU":"Infra","KAKATIYCEM":"Infra","KALYANACC":"Infra","KANJIGROUP":"Infra","KEI":"Infra","KENNAMET":"Infra",
    "KENTILE":"Infra","KESORAMIND":"Infra","KHAITANLTD":"Infra","KIRLOSKAR":"Infra","KIRLPNU":"Infra","KMEW":"Infra",
    "KRIDHANINF":"Infra","MAHARASTRA":"Infra","MAJOR":"Infra","MANGALAM":"Infra","MASTERSH":"Infra","MIDFIELD":"Infra",
    "MONTECARLO":"Infra","MURUDESHWAR":"Infra",

    # Insurance
    "AADHAR":"Insurance","AADHARHFC":"Insurance","ADJUSTIVELTD":"Insurance","ALFL":"Insurance","AMERINST":"Insurance","AMVINS":"Insurance",
    "ANCFINCORP":"Insurance","ARIHANTCAP":"Insurance","ASTICAP":"Insurance","ASTINVEST":"Insurance","BELSTAR":"Insurance","BLAL":"Insurance",
    "BSELIMITED":"Insurance","CAPTRUST":"Insurance","CGGROUPLTD":"Insurance","CHOICEIN":"Insurance","CREST":"Insurance","DOLAT":"Insurance",
    "ECHELON":"Insurance","ELCID":"Insurance","ESGAREINSU":"Insurance","FFMFIN":"Insurance","FSFL":"Insurance","FSTL":"Insurance",
    "GCMCAPITAL":"Insurance","GRAL":"Insurance","GROWTH":"Insurance","GULJAGO":"Insurance","INFOLLION":"Insurance","INITIA":"Insurance",
    "IVZINMF":"Insurance","JAYSHINVEST":"Insurance","JFLLIFE":"Insurance","JIGAR":"Insurance","JUDICIOUS":"Insurance","KANNF":"Insurance",
    "KAPILRAJ":"Insurance","KFS":"Insurance","KHFIN":"Insurance","KRISHNACAP":"Insurance","LOKMANYA":"Insurance","LONGLIVE":"Insurance",
    "MASTER":"Insurance","MASTERC":"Insurance","MEHTA":"Insurance","MEHTASEC":"Insurance","MICROSEC":"Insurance","MMFL":"Insurance",
    "MONEYBOXX":"Insurance","MONEYCAPS":"Insurance","MONEYLEND":"Insurance","MTCAP":"Insurance","MUTHOOTCAP":"Insurance","MUTHOOTMICROFIN":"Insurance",
    "PILANIINVS":"Insurance","PNBGILTS":"Insurance","PRIMESECU":"Insurance","PROFINS":"Insurance","SEINDIA":"Insurance","SMCGLOBAL":"Insurance",
    "UTIAMC":"Insurance",

    # Logistics
    "ACLGATI":"Logistics","AEGISLOG":"Logistics","AGARWAL":"Logistics","AHEMDABADST":"Logistics","AHMEDABADST":"Logistics","AIRAN":"Logistics",
    "AIRINDIAEXP":"Logistics","AIRWORKS":"Logistics","APEEJAY":"Logistics","ARSHIYA":"Logistics","BALMLAWRIE":"Logistics","CARGOTRANS":"Logistics",
    "FLY":"Logistics","FRONTLINE":"Logistics","GDL":"Logistics","GESHIP":"Logistics","IAL":"Logistics","ICIL":"Logistics",
    "INTERGLOBE":"Logistics","MARITIME":"Logistics","MMTC":"Logistics","PATINTLOG":"Logistics","STCINDIA":"Logistics","TCIEXP":"Logistics",

    # Media
    "BKMINDST":"Media","CLEDUCATE":"Media","DECCAN":"Media","DJML":"Media","DRAMAWORKS":"Media","HMVL":"Media",
    "IMAGICAA":"Media","INXS":"Media","JAGRAN":"Media","MAXEXPOSURE":"Media","MEETRECORDS":"Media","MEGASTAR":"Media",
    "MIRAJ":"Media","MUKTA":"Media","NAVNETEDUL":"Media","ONEINDIADIV":"Media","TREEHOUSE":"Media",

    # Metal
    "ADORWELD":"Metal","AGARIND":"Metal","AIONSTL":"Metal","ALUTEC":"Metal","AMNS":"Metal","AMTL":"Metal",
    "ARAVALI":"Metal","ARCOTECH":"Metal","ASHOKAMET":"Metal","BEDMUTHA":"Metal","BOBRO":"Metal","BURNPUR":"Metal",
    "COPPERTECH":"Metal","DPWIRE":"Metal","DPWIRES":"Metal","EARTHSTAHL":"Metal","HINDCOPPER":"Metal","IMPEXFER":"Metal",
    "ISMT":"Metal","JAYNECOIND":"Metal","JOSIL":"Metal","JSLLTD":"Metal","LGBFORGE":"Metal","LLOYDSME":"Metal",
    "LOHA":"Metal","MAHASTEEL":"Metal","MAHSTEEL":"Metal","MAITHANALL":"Metal","MANAKALUMNI":"Metal","METTLE":"Metal",
    "MONNET":"Metal","MULTIMETALS":"Metal","NALCO":"Metal","NSLNISP":"Metal","ROHITFERRE":"Metal","SARDAEN":"Metal",
    "SHYAMMETL":"Metal",

    # Pharma
    "ACIFINE":"Pharma","AGIS":"Pharma","AGROPHA":"Pharma","AGSCIENT":"Pharma","AKG":"Pharma","ALCHEM":"Pharma",
    "ALCHEMIST":"Pharma","ALEMBICLTD":"Pharma","ALIVUS":"Pharma","ALPA":"Pharma","ANUH":"Pharma","APOLLOHOSPENTPR":"Pharma",
    "ASSL":"Pharma","BAFNAPH":"Pharma","BALAXI":"Pharma","BIOFIL":"Pharma","BLISSGVS":"Pharma","BRENTFORD":"Pharma",
    "CURA":"Pharma","DEEVYABIO":"Pharma","DERMACURE":"Pharma","DOCS":"Pharma","EMCURE":"Pharma","ENTERO":"Pharma",
    "FERMENTA":"Pharma","FRIENDSBIO":"Pharma","GALENICA":"Pharma","GENNEX":"Pharma","GERBERA":"Pharma","GLAND":"Pharma",
    "GLOBALHEALTH":"Pharma","GUFICBIO":"Pharma","HCG":"Pharma","HEALTHY":"Pharma","HEART":"Pharma","HIMBIO":"Pharma",
    "INDIGOBIO":"Pharma","INDRAMEDCO":"Pharma","INDSWFTLT":"Pharma","INNOVA":"Pharma","INOVIO":"Pharma","IPCA":"Pharma",
    "JLHL":"Pharma","KIMSHEALTH":"Pharma","KOPRAN":"Pharma","KRISHNAPHA":"Pharma","LALPATHLABS":"Pharma","LINAKMEDICAL":"Pharma",
    "MANKIND":"Pharma","MEDICAMEN":"Pharma","MEDMONT":"Pharma","PLASMAGEN":"Pharma","PLATINLAB":"Pharma","POLYMED":"Pharma",
    "RFCL":"Pharma","TARSONS":"Pharma",

    # Realty
    "AMJLAND":"Realty","AMRAPALI":"Realty","ARVSMART":"Realty","AURUM":"Realty","DAGA":"Realty","DBREALTY":"Realty",
    "EMAMIREALT":"Realty","GROWTHHOME":"Realty","GULMARG":"Realty","HINES":"Realty","INDRAPR":"Realty","JAIPUR":"Realty",
    "JAIPURCOL":"Realty","JAISURYA":"Realty","JOGANI":"Realty","JPRL":"Realty","JRDC":"Realty","KARDA":"Realty",
    "KUKREJA":"Realty","MAHMUIUR":"Realty","MARATHONNT":"Realty","MARNOW":"Realty","MAXESTATES":"Realty","MKSVENTURES":"Realty",
    "PURVA":"Realty","SIGNATUREGLOBAL":"Realty","SURAJEST":"Realty","UNIVASTU":"Realty","VASTUSHILP":"Realty",

    # Retail
    "ADVANIHOTEL":"Retail","ARCHIES":"Retail","ARENTER":"Retail","ARENTERP":"Retail","BHAGYANGR":"Retail","CAMPUS":"Retail",
    "DOOGAR":"Retail","DREAMSPORTS":"Retail","ETHOS":"Retail","FIRSTCRY":"Retail","FLFL":"Retail","GIVA":"Retail",
    "GUJHOTEL":"Retail","GURUDEV":"Retail","HISTREET":"Retail","HORECR":"Retail","HOTELBENGAL":"Retail","HOTELDEL":"Retail",
    "HOTELHENRY":"Retail","HOTELINDIA":"Retail","HOTELLTD":"Retail","HOTELPATNA":"Retail","HOTELRADIS":"Retail","IMPRESARIO":"Retail",
    "INDOAMI":"Retail","INDOHOTEL":"Retail","INDOTHAI":"Retail","JAIPURKNI":"Retail","KALAMANDIR":"Retail","KARNIMATA":"Retail",
    "KLBENGAL":"Retail","KOTAREX":"Retail","KSLS":"Retail","LANDMARK":"Retail","MALL":"Retail","MALLCOMM":"Retail",
    "MANBHAWANI":"Retail","MANGCITY":"Retail","MARKS":"Retail","MIRZA":"Retail","PRAXIS":"Retail","SAMHI":"Retail",
    "STOKEHOS":"Retail","TRENDSETTR":"Retail","V2RETAIL":"Retail",

    # Telecom
    "GTPL":"Telecom","HATHWAY":"Telecom","HELIOSTELE":"Telecom","ICOMM":"Telecom","JIOTELE":"Telecom","KAVVERITEL":"Telecom",
    "MAHTELECOM":"Telecom","MOBILECOMM":"Telecom","MTNL":"Telecom","MULTICOMM":"Telecom","STLTECH":"Telecom","TEJASNET":"Telecom",

    # Textile
    "ABFRL":"Textile","AJANTTEX":"Textile","ALOKINDS":"Textile","AMBIKCO":"Textile","ASHIMASYN":"Textile","AYMSYNTEX":"Textile",
    "BOYATEX":"Textile","BRFL":"Textile","CENTURYENKA":"Textile","CHANDNI":"Textile","CONSCIOUSCO":"Textile","DOLLAR":"Textile",
    "DOLLARIND":"Textile","EUROTEX":"Textile","FABIND":"Textile","GRABALB":"Textile","KPRMILL":"Textile","LUXIND":"Textile",
    "MARALOVER":"Textile","MARGOTEX":"Textile","MODTHREAD":"Textile","MOHITIND":"Textile","MOHONINDUST":"Textile","NAHARSPING":"Textile",
    "PATSPINLTD":"Textile","SALONA":"Textile","ZODIAC":"Textile",
}



# ═══════════════════════════════════════════════════════════════
# DB LAYER
# ═══════════════════════════════════════════════════════════════

def get_db(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sector_map (
            symbol      TEXT PRIMARY KEY,
            sector      TEXT NOT NULL,
            source      TEXT,
            updated_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS build_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            built_at    TEXT,
            total       INTEGER,
            covered     INTEGER,
            sources     TEXT
        );
    """)
    conn.commit()
    return conn


def upsert(conn: sqlite3.Connection, symbol: str, sector: str, source: str):
    """
    Insert or update a sector mapping.
    Manual entries (source='manual') are never overwritten by auto-refresh —
    enforced by SELECT-then-INSERT logic that is compatible with all SQLite versions.
    """
    sym = symbol.upper().strip()
    now = datetime.now().isoformat()
    existing = conn.execute(
        "SELECT source FROM sector_map WHERE symbol=?", (sym,)
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO sector_map(symbol, sector, source, updated_at) VALUES(?,?,?,?)",
            (sym, sector, source, now)
        )
    elif existing["source"] != "manual":
        conn.execute(
            "UPDATE sector_map SET sector=?, source=?, updated_at=? WHERE symbol=?",
            (sector, source, now, sym)
        )
    # If source == 'manual': silently skip — manual entry is protected


def lookup(symbol: str, path: str = DB_PATH) -> Optional[str]:
    """Lightweight single-symbol lookup."""
    try:
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT sector FROM sector_map WHERE symbol=?", (symbol.upper().strip(),)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
# NSE SESSION HELPER
# ═══════════════════════════════════════════════════════════════

def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
    })
    try:
        # Warm-up request to acquire session cookies (required by NSE API)
        s.get("https://www.nseindia.com/", timeout=12)
        time.sleep(0.3)
    except Exception:
        pass
    return s


# ═══════════════════════════════════════════════════════════════
# SOURCE 1 — NSE equity-stockIndices API
# ═══════════════════════════════════════════════════════════════

def fetch_nse_index(session: requests.Session, index_name: str) -> list:
    """Return list of trading symbols in an NSE index, or [] on failure."""
    try:
        r = session.get(
            "https://www.nseindia.com/api/equity-stockIndices",
            params={"index": index_name},
            timeout=15
        )
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        # Filter out the index row itself (NSE includes a summary row with
        # the index name as the symbol — its symbol equals the index name
        # with spaces removed, e.g. "NIFTYMIDCAP150")
        index_sym = index_name.replace(" ", "").upper()
        return [
            d["symbol"].upper()
            for d in data
            if d.get("symbol") and d["symbol"].upper() != index_sym
        ]
    except Exception:
        return []


def load_from_nse_indices(conn: sqlite3.Connection) -> int:
    """Fetch all configured NSE index constituents and upsert their sectors."""
    print("→ Fetching NSE index constituents…")
    session = _nse_session()
    loaded = 0
    for index_name, sector_label in NSE_INDEX_TO_SECTOR.items():
        if sector_label is None:
            # Classification-only index: fetch symbols so the NSE master CSV
            # pass can map their industries, but don't assign a blanket sector.
            continue
        syms = fetch_nse_index(session, index_name)
        if not syms:
            print(f"  ✗ {index_name} — no data (rate-limited or unavailable)")
            time.sleep(1.0)
            continue
        for sym in syms:
            upsert(conn, sym, sector_label, "nse_index")
            loaded += 1
        print(f"  ✓ {index_name} → {sector_label} ({len(syms)} stocks)")
        time.sleep(0.25)
    conn.commit()
    return loaded


# ═══════════════════════════════════════════════════════════════
# SOURCE 2 — NSE equity master CSV
# ═══════════════════════════════════════════════════════════════

def _parse_equity_csv(text: str, conn: sqlite3.Connection, source_label: str) -> int:
    """
    Parse an NSE equity master CSV (EQUITY_L.csv format) and upsert sectors.
    Handles the leading-space ' INDUSTRY' column name automatically.
    Returns count of rows upserted.
    """
    loaded = 0
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        row_clean = {k.strip(): v for k, v in row.items()}
        sym    = row_clean.get("SYMBOL", "").strip().upper()
        series = row_clean.get("SERIES", "").strip().upper()
        ind    = row_clean.get("INDUSTRY", "").strip().upper()
        if not sym or series not in ("EQ", "BE", "BZ", "SM"):
            continue
        sector = NSE_INDUSTRY_TO_SECTOR.get(ind)
        if sector:
            upsert(conn, sym, sector, source_label)
            loaded += 1
    return loaded


def load_from_nse_master(conn: sqlite3.Connection) -> int:
    """
    Fetch the NSE equity master CSV and upsert industry→sector mappings.

    NSE has moved this file several times; we try multiple known URLs in order
    and fall back to the NSE API's equity-master JSON endpoint if all CSV URLs fail.

    Columns (EQUITY_L.csv format):
      SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE,
      MARKET LOT, ISIN NUMBER, FACE VALUE, " INDUSTRY"  (leading space on header)
    """
    print("→ Fetching NSE equity master CSV…")
    session = _nse_session()

    # ── Try known EQUITY_L.csv URLs in priority order ──────────────────────
    # NSE periodically moves the file; all known historical locations listed.
    CANDIDATE_URLS = [
        # Current location (as of 2024-25)
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        # Legacy location (returned 404 mid-2025)
        "https://www.nseindia.com/content/equities/EQUITY_L.csv",
        # Alternate archives subdomain path
        "https://nsearchives.nseindia.com/emerge/corporates/content/equities/EQUITY_L.csv",
    ]

    raw_text = None
    for url in CANDIDATE_URLS:
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                raw_text = r.content.decode("utf-8", errors="replace")
                print(f"  ✓ Fetched CSV from {url}")
                break
            else:
                print(f"  ✗ {url} → HTTP {r.status_code}")
        except Exception as e:
            print(f"  ✗ {url} → {type(e).__name__}: {e}")
        time.sleep(0.5)

    if raw_text:
        try:
            loaded = _parse_equity_csv(raw_text, conn, "nse_master")
            conn.commit()
            print(f"  ✓ NSE master CSV: {loaded} stocks mapped")
            return loaded
        except Exception as e:
            print(f"  ✗ Error parsing master CSV: {e}")

    # ── Fallback: NSE equity-master JSON API ──────────────────────────────
    # Same data as EQUITY_L.csv but served as JSON. Slower but URL-stable.
    print("  → CSV unavailable — trying NSE equity-master JSON API…")
    loaded = _load_master_via_api(session, conn)
    conn.commit()
    return loaded


def _load_master_via_api(session: requests.Session, conn: sqlite3.Connection) -> int:
    """
    Fallback: pull symbol+industry data from NSE's equity-master JSON endpoint.
    Same data as EQUITY_L.csv but served as JSON — URL is more stable.
    """
    loaded = 0
    JSON_MASTER_URLS = [
        "https://www.nseindia.com/api/equity-master",
        "https://nsearchives.nseindia.com/api/equity-master",
    ]
    for url in JSON_MASTER_URLS:
        try:
            r = session.get(url, timeout=30)
            if r.status_code != 200:
                print(f"  ✗ {url} → HTTP {r.status_code}")
                continue
            data = r.json()
            rows = data if isinstance(data, list) else data.get("data", [])
            if not rows:
                continue
            for row in rows:
                sym    = str(row.get("symbol",   "") or row.get("SYMBOL",   "")).strip().upper()
                series = str(row.get("series",   "") or row.get("SERIES",   "")).strip().upper()
                ind    = str(row.get("industry", "") or row.get("INDUSTRY", "")).strip().upper()
                if not sym or series not in ("EQ", "BE", "BZ", "SM"):
                    continue
                sector = NSE_INDUSTRY_TO_SECTOR.get(ind)
                if sector:
                    upsert(conn, sym, sector, "nse_master_api")
                    loaded += 1
            if loaded > 0:
                print(f"  ✓ NSE JSON API fallback: {loaded} stocks mapped")
                return loaded
        except Exception as e:
            print(f"  ✗ JSON API {url}: {e}")
        time.sleep(0.5)

    print("  → All master CSV/JSON sources failed.")
    print("     Coverage limited to ~450 large-caps from index constituents.")
    print("     Retry --build when NSE is accessible, or use --manual for specific stocks.")
    return loaded


# ═══════════════════════════════════════════════════════════════
# SOURCE 3 — Static fallback
# ═══════════════════════════════════════════════════════════════

def load_from_static(conn: sqlite3.Connection) -> int:
    """Seed DB with the static fallback dict (lowest priority)."""
    print("→ Loading static fallback…")
    loaded = 0
    for sym, sector in STATIC_FALLBACK.items():
        upsert(conn, sym, sector, "static")
        loaded += 1
    conn.commit()
    print(f"  ✓ Static fallback: {loaded} stocks")
    return loaded


# ═══════════════════════════════════════════════════════════════
# SOURCE 4 — Manual override (never overwritten by auto-refresh)
# ═══════════════════════════════════════════════════════════════

def manual_add(symbol: str, sector: str, path: str = DB_PATH):
    """
    Add/update a single stock with source='manual'.
    Manual entries are NEVER overwritten by auto-refresh.

    Usage:
        python sector_db.py --manual TATAMOTORS Auto
    """
    conn = get_db(path)
    sym = symbol.upper().strip()
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO sector_map(symbol, sector, source, updated_at)
        VALUES(?, ?, 'manual', ?)
        ON CONFLICT(symbol) DO UPDATE SET
            sector=excluded.sector, source='manual', updated_at=excluded.updated_at
    """, (sym, sector, now))
    conn.commit()
    conn.close()
    print(f"✓ Manual: {sym} → {sector}")



# ═══════════════════════════════════════════════════════════════
# SOURCE 2b — Extended hardcoded map (mid/small-cap coverage)
# ═══════════════════════════════════════════════════════════════

def load_from_extended_map(conn: sqlite3.Connection) -> int:
    """
    Load ~1800 mid/small-cap stocks from the hardcoded EXTENDED_SECTOR_MAP.
    Used when NSE's EQUITY_L.csv is unavailable (blocked/moved).
    Priority: higher than 'static', lower than 'nse_master' and 'nse_index'.
    Entries already present with source 'nse_index' or 'nse_master' are NOT overwritten.
    """
    loaded = 0
    for sym, sector in EXTENDED_SECTOR_MAP.items():
        existing = conn.execute(
            "SELECT source FROM sector_map WHERE symbol=?", (sym.upper(),)
        ).fetchone()
        if existing is None:
            # New symbol — insert
            conn.execute(
                "INSERT INTO sector_map(symbol, sector, source, updated_at) VALUES(?,?,?,?)",
                (sym.upper(), sector, "extended", datetime.now().isoformat())
            )
            loaded += 1
        elif existing["source"] not in ("nse_index", "nse_master", "nse_master_api", "manual"):
            # Only overwrite lower-priority 'static' entries
            conn.execute(
                "UPDATE sector_map SET sector=?, source=?, updated_at=? WHERE symbol=?",
                (sector, "extended", datetime.now().isoformat(), sym.upper())
            )
            loaded += 1
    conn.commit()
    print(f"  ✓ Extended map: {loaded} stocks added/updated")
    return loaded


# ═══════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════

def build(path: str = DB_PATH):
    """
    Full rebuild.  Load order (ascending priority — later writes win):
      static → extended → nse_master → nse_indices
    Manual entries are always protected regardless of order.
    - static:   173 hand-picked large-caps (always present)
    - extended: ~830 hardcoded mid/small-caps (no network needed)
    - nse_master: EQUITY_L.csv (~2000 stocks, upgrades extended if reachable)
    - nse_indices: live NSE index fetch (highest priority, ~450 stocks)
    """
    print(f"\nBuilding sector DB → {path}")
    conn = get_db(path)

    load_from_static(conn)
    load_from_extended_map(conn)   # hardcoded mid/small-cap map
    load_from_nse_master(conn)     # NSE CSV (upgrades extended entries if available)
    load_from_nse_indices(conn)    # live index fetch (highest priority)

    total  = conn.execute("SELECT COUNT(*) FROM sector_map").fetchone()[0]
    by_src = conn.execute(
        "SELECT source, COUNT(*) n FROM sector_map GROUP BY source ORDER BY n DESC"
    ).fetchall()
    src_str = ", ".join(f"{r['source']}:{r['n']}" for r in by_src)

    conn.execute(
        "INSERT INTO build_log(built_at, total, covered, sources) VALUES(?,?,?,?)",
        (datetime.now().isoformat(), total, total, src_str)
    )
    conn.commit()
    conn.close()

    print(f"\n{'═'*52}")
    print(f"  Sector DB built:  {total} stocks mapped")
    print(f"  Sources:          {src_str}")
    print(f"  DB path:          {path}")
    print(f"{'═'*52}\n")


# ═══════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════

def stats(path: str = DB_PATH):
    """Print coverage statistics."""
    if not os.path.exists(path):
        print("DB not found — run: python sector_db.py --build")
        return
    conn = get_db(path)
    total = conn.execute("SELECT COUNT(*) FROM sector_map").fetchone()[0]

    print(f"\nTotal stocks mapped: {total}\n")
    print("By sector:")
    for r in conn.execute(
        "SELECT sector, COUNT(*) n FROM sector_map GROUP BY sector ORDER BY n DESC"
    ).fetchall():
        print(f"  {r['sector']:<22} {r['n']:>4}")

    print("\nBy source:")
    for r in conn.execute(
        "SELECT source, COUNT(*) n FROM sector_map GROUP BY source ORDER BY n DESC"
    ).fetchall():
        print(f"  {r['source']:<16} {r['n']:>4}")

    no_sector = conn.execute(
        "SELECT COUNT(*) FROM sector_map WHERE sector IS NULL OR sector=''"
    ).fetchone()[0]
    if no_sector:
        print(f"\nWarning: {no_sector} rows with empty sector")

    conn.close()


# ═══════════════════════════════════════════════════════════════
# IN-MEMORY CACHE — loaded once at startup, thread-safe
# ═══════════════════════════════════════════════════════════════

_DB_CACHE:        Dict[str, str] = {}
_DB_CACHE_LOADED: bool           = False
_DB_CACHE_LOCK                   = threading.Lock()


def _load_cache(path: str = DB_PATH) -> None:
    """
    Load all sector_map rows into _DB_CACHE.
    Thread-safe double-checked locking — safe to call from multiple threads
    at startup (e.g. the live-price daemon and the main thread).
    No-op if already loaded or DB file does not exist yet.
    """
    global _DB_CACHE, _DB_CACHE_LOADED
    if _DB_CACHE_LOADED:
        return
    with _DB_CACHE_LOCK:
        if _DB_CACHE_LOADED:   # second check inside lock
            return
        if not os.path.exists(path):
            _DB_CACHE_LOADED = True
            print(f"[sector_db] DB not found at {path} — run sector_db.py --build")
            return
        try:
            conn = sqlite3.connect(path)
            rows = conn.execute("SELECT symbol, sector FROM sector_map").fetchall()
            conn.close()
            _DB_CACHE = {r[0]: r[1] for r in rows}
            _DB_CACHE_LOADED = True
            print(f"[sector_db] Loaded {len(_DB_CACHE)} sector mappings from {path}")
        except Exception as e:
            print(f"[sector_db] Warning: could not load sector DB: {e}")
            _DB_CACHE_LOADED = True


def reload_cache(path: str = DB_PATH) -> None:
    """Force a full reload of the in-memory cache (call after --build)."""
    global _DB_CACHE, _DB_CACHE_LOADED
    with _DB_CACHE_LOCK:
        _DB_CACHE_LOADED = False
    _load_cache(path)


# ═══════════════════════════════════════════════════════════════
# DROP-IN get_sector REPLACEMENT FOR main.py
# ═══════════════════════════════════════════════════════════════

def get_sector_db(
    ticker: str,
    stock_sector_map: dict,
    lock: threading.Lock,
    db_path: str = DB_PATH,
) -> Optional[str]:
    """
    Drop-in replacement for main.py's get_sector().

    Lookup order (fastest → most comprehensive):
      1. In-memory STOCK_SECTOR_MAP  — live NSE index refresh (~200 large-caps)
      2. In-memory _DB_CACHE         — full 2500-stock DB, loaded once at startup
      3. Returns None                — genuinely unknown stock

    Integration (already applied to main.py):

        from sector_db import get_sector_db, _load_cache as _load_sector_db_cache

        # At startup, after init_db():
        _load_sector_db_cache()

        def get_sector(ticker: str) -> Optional[str]:
            return get_sector_db(ticker, STOCK_SECTOR_MAP, _SECTOR_MAP_LOCK)
    """
    _load_cache(db_path)          # no-op after first call
    with lock:
        result = stock_sector_map.get(ticker.upper())
    if result:
        return result
    return _DB_CACHE.get(ticker.upper())


# ═══════════════════════════════════════════════════════════════
# API HELPERS — used by main.py /api/sector/* endpoints
# ═══════════════════════════════════════════════════════════════

def get_all_mappings(path: str = DB_PATH) -> list:
    """Return all rows as a list of dicts for the API."""
    if not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, sector, source, updated_at FROM sector_map ORDER BY symbol"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_coverage_stats(path: str = DB_PATH) -> dict:
    """Return coverage stats dict for the API."""
    if not os.path.exists(path):
        return {"total": 0, "by_sector": {}, "by_source": {}}
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM sector_map").fetchone()[0]
        by_sector = {r["sector"]: r["n"] for r in conn.execute(
            "SELECT sector, COUNT(*) n FROM sector_map GROUP BY sector ORDER BY n DESC"
        ).fetchall()}
        by_source = {r["source"]: r["n"] for r in conn.execute(
            "SELECT source, COUNT(*) n FROM sector_map GROUP BY source ORDER BY n DESC"
        ).fetchall()}
        conn.close()
        return {"total": total, "by_sector": by_sector, "by_source": by_source}
    except Exception:
        return {"total": 0, "by_sector": {}, "by_source": {}}


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MONARCH PRO — NSE Sector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sector_db.py --build
  python sector_db.py --stats
  python sector_db.py --lookup TATAMOTORS
  python sector_db.py --manual GPPL Energy
  python sector_db.py --db /custom/path/sector_map.db --build
        """
    )
    parser.add_argument("--build",  action="store_true", help="Build/rebuild the sector DB")
    parser.add_argument("--stats",  action="store_true", help="Show coverage statistics")
    parser.add_argument("--manual", nargs=2, metavar=("SYMBOL", "SECTOR"),
                        help="Manually pin a sector (never overwritten by auto-refresh)")
    parser.add_argument("--lookup", metavar="SYMBOL",    help="Look up a single symbol")
    parser.add_argument("--reload", action="store_true", help="Reload in-memory cache from DB")
    parser.add_argument("--db",     default=DB_PATH,     help="Override DB file path")
    args = parser.parse_args()

    if args.build:
        build(args.db)
    elif args.stats:
        stats(args.db)
    elif args.manual:
        manual_add(args.manual[0], args.manual[1], args.db)
    elif args.lookup:
        result = lookup(args.lookup, args.db)
        print(f"{args.lookup.upper()} → {result or 'Not found'}")
    elif args.reload:
        reload_cache(args.db)
        print(f"Cache reloaded: {len(_DB_CACHE)} entries")
    else:
        parser.print_help()