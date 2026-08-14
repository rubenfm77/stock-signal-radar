"""
Curated ticker universe across US, European, and Spanish markets.
Kept intentionally small (~60-80 names) for fast iteration; expand once
the pipeline is validated. yfinance ticker suffix conventions:
  - Spain (Madrid):   .MC
  - Germany (Xetra):  .DE
  - France (Paris):   .PA
  - Netherlands:      .AS
  - Italy (Milan):    .MI
  - US:               no suffix
"""

IBEX35 = [
    "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC",
    "FER.MC", "AMS.MC", "CLNX.MC", "ELE.MC", "NTGY.MC", "CABK.MC",
    "AENA.MC", "MAP.MC", "ACS.MC", "GRF.MC", "SAB.MC", "IAG.MC",
    "COL.MC", "MRL.MC", "ANA.MC", "ACX.MC", "BKT.MC", "ENG.MC",
    "FDR.MC", "IDR.MC", "LOG.MC", "MEL.MC", "PHM.MC", "PUIG.MC",
    "RED.MC", "ROVI.MC", "SCYR.MC", "SLR.MC", "UNI.MC",
]

EUROSTOXX_SAMPLE = [
    "ASML.AS", "SAP.DE", "SIE.DE", "ALV.DE", "MC.PA", "OR.PA",
    "TTE.PA", "AIR.PA", "SU.PA", "DTE.DE", "BAS.DE", "ADS.DE",
    "ENEL.MI", "ISP.MI", "STLAM.MI", "BAYN.DE", "BMW.DE", "VOW3.DE",
    "MBG.DE", "RWE.DE", "DB1.DE", "IFX.DE", "MUV2.DE", "DPW.DE",
    "BNP.PA", "SAN.PA", "CS.PA", "DG.PA", "RMS.PA", "KER.PA",
    "ENI.MI", "RACE.MI", "UCG.MI", "ASM.AS", "PHIA.AS", "INGA.AS",
]

US_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "PG", "MA", "HD", "COST", "AVGO",
    "ADBE", "NFLX", "CRM", "AMD", "INTC", "PEP", "KO",
    "WMT", "BAC", "DIS", "CSCO", "ORCL", "ABBV", "MRK", "PFE",
    "TMO", "ABT", "ACN", "MCD", "NKE", "TXN", "QCOM", "LIN",
    "HON", "UPS", "IBM", "GE", "CAT", "BA", "GS", "MS", "SBUX", "PYPL",
]

UNIVERSE = {
    "IBEX35 (España)": IBEX35,
    "EuroStoxx (muestra)": EUROSTOXX_SAMPLE,
    "US (muestra S&P/Nasdaq)": US_SAMPLE,
}


def all_tickers() -> list[str]:
    seen, out = set(), []
    for group in UNIVERSE.values():
        for t in group:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out
