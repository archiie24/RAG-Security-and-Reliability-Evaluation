from evaluation.redteam import run as run_redteam
from evaluation.reliability import run as run_reliability

if __name__=='__main__':
    df,summary=run_redteam(); print('\nRED TEAM SUMMARY\n',summary.to_string(index=False)); df.to_csv('redteam_results.csv',index=False)
    rel=run_reliability(); print('\nRELIABILITY RESULTS\n',rel.to_string(index=False)); rel.to_csv('reliability_results.csv',index=False)
