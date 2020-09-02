from numpy import dot
from numpy.linalg import norm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

maxID = 121
maxGame = 11

def cos_sim(v1, v2): 
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm     
    
    return similarity

def SqlDFLoad(file, execute):
  # SQLite DB 연결
  conn = sqlite3.connect("content/" + str(file))

  df = pd.read_sql(execute, conn)

  conn.close()

  return df

def SqlDFSave(file, tableDF, tableName):
  # SQLite DB 연결
  conn = sqlite3.connect("content/" + str(file))

  tableDF.to_sql(tableName, conn, if_exists='append', index=False)

  conn.close()

def AllPlayerPosCos(dbName):
    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = SqlDFLoad(dbName, "select ID, gameNum, step, xPos, yPos from ML")
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos from ML")

    P_gameCount = PlayerTable['gameNum'][len(PlayerTable)-2]
    saveDBname = dbName.split('playerData')
    saveDBname = saveDBname[1].split('.db')
    saveDBname = int(saveDBname[0])

    idCount = 21
    gameCount = 11

    cosDF = list()
    avgDF = list()

    for Game in range(1, P_gameCount+1):
        playerFilterTable = PlayerTable[PlayerTable['gameNum'] == Game]
        for ID in range(1, idCount):
            for aiGame in range(1, gameCount):
                # ai 필터링
                aiFilter = AiTable[AiTable['ID'] == str(ID)]
                aiFilter = aiFilter[aiFilter['gameNum'] == aiGame]

                playerTablePos = playerFilterTable[['gameNum', 'step', 'xPos', 'yPos']]
                aiFilterPos = aiFilter[['ID', 'gameNum', 'xPos', 'yPos']]

                playerTablePos.columns = ['P_Game', 'Step', 'P_xPos', 'P_yPos']
                aiFilterPos.columns = ['A_ID', 'A_Game', 'A_xPos', 'A_yPos']

                # 두 포지션을 합치고 빈자리를 0으로 만듬
                contactDF = pd.concat([playerTablePos, aiFilterPos], axis=1)
                contactDF = contactDF.fillna(1)

                # 유사도 계산
                contactDF['Cos'] = cos_sim(contactDF['P_yPos'].values, contactDF['A_yPos'].values)

                contactDF['P_ID'] = str(saveDBname)
                contactDF['P_Game'] = Game
                contactDF['A_ID'] = str(ID)
                contactDF['A_Game'] = aiGame

                cosDF.append(contactDF)

                avg = np.mean(contactDF['Cos'].values)
                # 데이터 P - ID, Game, A - ID, Game, Cos
                appendAvgDF = pd.DataFrame(data=[(str(saveDBname), Game, str(ID), aiGame, avg)],columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                avgDF.append(appendAvgDF)
            #print('플레이어 좌표데이터 Ai ID 처리: ' + str(ID)+ '/' + str(idCount))
        print('플레이어 좌표데이터 처리: ' + str(Game)+ '/' + str(gameCount))

    cosDF = pd.concat(cosDF)
    avgDF = pd.concat(avgDF)
    # 정렬
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    avgID_DF = list()
    # 최종
    for ID in range(1, idCount):
        aiFilter = avgDF[avgDF['A_ID'] == str(ID)]
        aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
        aiFilter = aiFilter.drop_duplicates(['P_Game'])
        avg = np.mean(aiFilter['Cos'].values)
        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
        avgID_DF.append(appendAvgDF)

    avgID_DF = pd.concat(avgID_DF)
    avgID_DF = avgID_DF.sort_values(by=['Cos'], ascending=False, axis=0)
    
    print('포지션 처리 완료')

    # 모든 작업이 끝나면 저장
    SqlDFSave('saveSqlData.db', cosDF, 'PosCos')
    SqlDFSave('saveSqlData.db', avgDF, 'PosAvg')
    SqlDFSave('saveSqlData.db', avgID_DF, 'PosIDAvg')

def OverPlayerCos(dbName):    
    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = SqlDFLoad(dbName, "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")

    P_gameCount = PlayerTable['gameNum'][len(PlayerTable)-2]
    idCount = 21
    gameCount = 11
    saveDBname = dbName.split('playerData')
    saveDBname = saveDBname[1].split('.db')
    saveDBname = int(saveDBname[0])

    # 일단 두개다 필터링
    P_Filter = OverFilter(PlayerTable)
    A_Filter = OverFilter(AiTable)

    playerTablePos = P_Filter[['gameNum', 'step', 'xPos', 'yPos']]
    aiFilterPos = A_Filter[['ID', 'gameNum', 'step', 'xPos', 'yPos']]

    playerTablePos.columns = ['P_Game', 'P_Step', 'P_xPos', 'P_yPos']
    aiFilterPos.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos']

    # 형변환
    playerOver = sparse.csr_matrix(P_Filter[['xPos', 'yPos']].values)
    aiOver = sparse.csr_matrix(A_Filter[['xPos', 'yPos']].values)

    # 유사도 계산
    similarity_simple_pair = cosine_similarity(playerOver, aiOver)
                
    tableDF = pd.DataFrame(similarity_simple_pair)

    playerTablePos.reset_index(drop=True, inplace=True)
    aiFilterPos.reset_index(drop=True, inplace=True)

    cosDF = list()

    # 유사도 계산 된 것을 정렬한다.
    for i in tableDF.index:
        input = tableDF.iloc[i]
        inDF = pd.concat([aiFilterPos, input], axis=1)
        inDF.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']
        inDF['P_ID'] = str(saveDBname)
        for col in playerTablePos.columns:
            inDF[col] = playerTablePos.get_value(i, col)
        # 정렬
        inDF.sort_values(by=['Cos'], ascending=False, axis=0)
        cosDF.append(inDF)
    
    cosDF = pd.concat(cosDF)
    cosDF = cosDF[['P_ID', 'P_Game', 'P_Step', 'P_xPos', 'P_yPos', 'A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']]

    avgDF = list()
    # 이제 검사된 값으로 반대로 가장 비슷한 학습량을 찾아냄
    # 평균을 계산할때 Ai와 플레이어의 횟수를 검사한다.
    for ID in range(1, idCount):
        # 먼저 필터한다.
        A_filterDF = cosDF[cosDF['A_ID'] == str(ID)]
        # 유사도 순으로 정렬1
        A_filterDF = A_filterDF.sort_values(by=['Cos'], ascending=False, axis=0)
        # 정렬 된 유사도 중 1등만 빼고 삭제한다.
        A_filterDF = A_filterDF.drop_duplicates(['P_Step'])
        # 플레이더 데이터가 AI 데이터보다 적을 경우
        aiFilter = aiFilterPos[aiFilterPos['A_ID'] == str(ID)]
        lenP = len(playerTablePos)
        lenA = len(aiFilter)
        avg = 0
        if lenP > lenA:
            # 정렬을 반대로 바꾼다
            A_filterDF = A_filterDF.sort_values(by=['Cos'], ascending= True, axis=0)
            plusNum = lenP - lenA
            for i in range(0, plusNum):
                A_filterDF['Cos'].values[i] = 0
            avg = np.mean(A_filterDF['Cos'].values)
        elif lenP < lenA:
            # 그 차만큼 허수를 생성
            plusNum = lenA - lenP
            plusArray = np.zeros(plusNum)
            orginArray = A_filterDF['Cos'].values
            conNP = np.concatenate((orginArray, plusArray), axis=0)
            avg = np.mean(conNP)
        else:
            avg = np.mean(A_filterDF['Cos'].values)
        
        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
        avgDF.append(appendAvgDF)
    
    avgDF = pd.concat(avgDF)
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('죽음 처리 완료')
    # 모든 작업이 끝나면 DF로 변환 후
    SqlDFSave('saveSqlData.db', cosDF, 'OverCos')
    SqlDFSave('saveSqlData.db', avgDF, 'OverAvg')

def OverFilter(OverDF):
    maxID = OverDF['ID'][len(OverDF)-2]
    maxGame = OverDF['gameNum'][len(OverDF)-2]
    inputList = list()
    inDF = pd.DataFrame({'A' : []})
    for ID in range(1, int(maxID)+1):
        filterDF =  OverDF[OverDF['ID'] == str(ID)]
        for game in range(1, maxGame+1):
            filterDF2 =  filterDF[filterDF['gameNum'] == game]
            if len(filterDF2) != 0:
                # 정렬
                filterDF2 = filterDF2.sort_values(by=['step'], ascending=False, axis=0)
                #filterDF2.reset_index(drop=True, inplace=True)
                #index = filterDF2.overStep.idxmax()
                appendDF = filterDF2.iloc[0:1]
                inputList.append(appendDF)
    inDF = pd.concat(inputList)
    return inDF

def JumpFilter(jumpDF, col):
    for i in jumpDF.index:
        if i+1 <= len(jumpDF.index)-1:
            # 인덱스, 컬럼
            value1 = jumpDF.get_value(i, col) +1 
            value2 = jumpDF.get_value(i+1, col)
            if value1 == value2:
                jumpDF.set_value(i, col, 0)
    
    return jumpDF[jumpDF[col] > 0]

def JumpPlayerCos(dbName):
    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = SqlDFLoad(dbName, "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")
    
    P_gameCount = PlayerTable['gameNum'][len(PlayerTable)-2]
    idCount = 21
    gameCount = 11
    saveDBname = dbName.split('playerData')
    saveDBname = saveDBname[1].split('.db')
    saveDBname = int(saveDBname[0])

    # 일단 두개다 필터링
    P_Filter = JumpFilter(PlayerTable, 'JumpStep')
    A_Filter = JumpFilter(AiTable, 'JumpStep')

    playerJump = sparse.csr_matrix(P_Filter[['step', 'xPos', 'yPos']].values)
    aiJump = sparse.csr_matrix(A_Filter[['step', 'xPos', 'yPos']].values)
    # 유사도 계산
    similarity_simple_pair = cosine_similarity(playerJump, aiJump)
                
    tableDF = pd.DataFrame(similarity_simple_pair)

    playerTablePos = P_Filter[['gameNum', 'step', 'xPos', 'yPos']]
    aiFilterPos = A_Filter[['ID', 'gameNum', 'step', 'xPos', 'yPos']]

    playerTablePos.columns = ['P_Game', 'P_Step', 'P_xPos', 'P_yPos']
    aiFilterPos.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos']

    # 계산된 Cos를 저장될 데이터로 수정한다.
    A_columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos']
    P_columns = ['P_Game', 'P_Step', 'P_xPos', 'P_yPos']
    
    playerTablePos.reset_index(drop=True, inplace=True)
    aiFilterPos.reset_index(drop=True, inplace=True)

    cosDF = list()
    cosFirstDF = list()

    #라벨링
    for i in tableDF.index:
        input = tableDF.iloc[i]
        inDF = pd.concat([aiFilterPos, input], axis=1)
        inDF.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']
        inDF['P_ID'] = saveDBname
        for col in playerTablePos.columns:
            inDF[col] = playerTablePos.get_value(i, col)
        # 정렬
        inDF = inDF.sort_values(by=['Cos'], ascending=False, axis=0)
        inDF = inDF[['P_ID', 'P_Game', 'P_Step', 'P_xPos', 'P_yPos', 'A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']]
        # 모든 정보를 다 저장하는 DF
        cosDF.append(inDF)        
    
    cosDF = pd.concat(cosDF)
    avgDF = list()

    # 검사한 것에서 각 게임마다 점프 검사한 것의 평균
    for Game in range(1, P_gameCount):
        # 게임
        playerFilter = cosDF[cosDF['P_Game'] == Game]
        # 실제 같은 step이 하나도록 수정한다.
        for ID in range(1, idCount):
            aiFilter = playerFilter[playerFilter['A_ID'] == str(ID)]
            for aiGame in range(1, gameCount):
                aiFilter2 = aiFilter[aiFilter['A_Game'] == aiGame]
                aiFilter2 = aiFilter2.sort_values(by=['Cos'], ascending=False, axis=0)
                aiFilter2 = aiFilter2.drop_duplicates(['P_Step'])
                
                # 플레이더 데이터가 AI 데이터보다 적을 경우
                aiFiltering = aiFilterPos[aiFilterPos['A_ID'] == str(ID)]
                aiFiltering = aiFiltering[aiFiltering['A_Game'] == aiGame]
                playerFiltering = playerTablePos[playerTablePos['P_Game'] == Game]
                lenP = len(playerFiltering)
                lenA = len(aiFiltering)
                avg = 0
                if lenP > lenA:
                    # 정렬을 반대로 바꾼다
                    aiFilter2 = aiFilter2.sort_values(by=['Cos'], ascending= True, axis=0)
                    plusNum = lenP - lenA
                    for i in range(0, plusNum):
                        aiFilter2['Cos'].values[i] = 0
                    avg = np.mean(aiFilter2['Cos'].values)
                elif lenP < lenA:
                    # 그 차만큼 허수를 생성
                    plusNum = lenA - lenP
                    plusArray = np.zeros(plusNum)
                    orginArray = aiFilter2['Cos'].values
                    conNP = np.concatenate((orginArray, plusArray), axis=0)
                    avg = np.mean(conNP)
                else:
                    avg = np.mean(aiFilter2['Cos'].values)

                appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), Game, str(ID), aiGame, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                avgDF.append(appendAvgDF)

    avgDF = pd.concat(avgDF)
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    avgID_DF = list()
    # 최종
    for ID in range(1, idCount):
        aiFilter = avgDF[avgDF['A_ID'] == str(ID)]
        aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
        aiFilter = aiFilter.drop_duplicates(['P_Game'])
        avg = np.mean(aiFilter['Cos'].values)
        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
        avgID_DF.append(appendAvgDF)

    avgID_DF = pd.concat(avgID_DF)
    avgID_DF = avgID_DF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('점프 처리 완료')

    # 모든 작업이 끝나면 DF로 변환 후
    SqlDFSave('saveSqlData.db', cosDF, 'JumpCos')
    SqlDFSave('saveSqlData.db', avgDF, 'JumpAvg')
    SqlDFSave('saveSqlData.db', avgID_DF, 'JumpIDAvg')

def Total(P_ID):
    OverDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from OverAvg")
    JumpDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from JumpIDAvg")
    CosDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from PosIDAvg")

    totalDF = list()

    idCount = 21
    P_OverFilter = OverDF[OverDF['P_ID'] == str(P_ID)]
    P_JumpFilter = JumpDF[JumpDF['P_ID'] == str(P_ID)]
    P_CosFilter = CosDF[CosDF['P_ID'] == str(P_ID)]
    for ID in range(1, idCount):
        OverFilter = OverDF[OverDF['A_ID'] == str(ID)]
        JumpFilter = JumpDF[JumpDF['A_ID'] == str(ID)]
        CosFilter = CosDF[CosDF['A_ID'] == str(ID)]
        conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
        avg = np.mean(conDF['Cos'].values)
        appendAvgDF =  pd.DataFrame(data=[(str(P_ID), ID, avg)], columns = ['P_ID', 'A_ID', 'Cos'])
        totalDF.append(appendAvgDF)

    totalDF = pd.concat(totalDF)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('종합 완료')
    SqlDFSave('saveSqlData.db', totalDF, 'TotalAvg')

### 죽음 데이터
#OverPlayerCos("playerData1.db")

### 점프
JumpPlayerCos("playerData1.db")

### 포지션
#AllPlayerPosCos("playerData1.db")

### 종합
#Total('1')