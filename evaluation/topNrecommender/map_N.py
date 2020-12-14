# df_actual has columns user_id,item_id
# df_predicted has columns user_id,item_id, score
# N= number of product recommendation given to each user
# m=number of unique relevant items for user from test data df_actual

import pandas as pd

def is_pred_relevant(df_actual,user_id,prediction):
    relevant_items=list(df_actual[df_actual["user_id"]==user_id]["item_id"].unique())
    if prediction in relevant items:
        return True
    return False
    
def map_N(df_actual,df_predicted,N):
    
    map_N=0
    for user in df_predicted.user_id.unique():
        
        m=df_actual[df_actual["user_id"]==user]["item_id"].nunique()
        is_superactive_user=False  # m>>N then true
        denom=min(m,N) if is_superactive_user else m
        
        df_user_top_N=df_predicted[df_predicted["user_id"]==user].sort_values(
                                                            by="score",ascending=False
                                                            ).head(N).reset_index(drop=True)
        df_user_top_N["relevant"]=df_user_top_N.apply(lambda x:is_pred_relevant(
                                                            df_actual,x[user_id],x[item_id]),axis=1)
        df_user_top_N["cum_relevants"]= df_user_top_N["relevant"].cumsum(axis=0)
        df_user_top_N["iterator"]=df_user_top_N.index+1
        df_user_top_N["precision_at_k"]=df_user_top_N.["cum_relevants"]/df_user_top_N["iterator"]
        
        Avg_precision=df_user_top_N["precision_at_k"].sum()/denom
        
        map_N+=Avg_precision
      
    map_N=map_N/df_predicted.user_id.nunique()
    
    return map_N