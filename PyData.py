from numpy import dot
from numpy.linalg import norm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
#import csv

P_maxID = 121
P_maxGame = 11
A_maxID = 121
A_maxGame = 11
A_intervalID = 2

# 0 PtoA, 1 PtoP, 2 AtoA

def cos_sim(v1, v2): 
    return dot(v1, v2)/(norm(v1)*norm(v2))

def SqlDFLoad(file, execute):
  # SQLite DB 연결
  conn = sqlite3.connect("content/" + str(file))

  df = pd.read_sql(execute, conn)

  conn.close()

  return df

def SqlDFSave(file, tableDF, tableName):
  # SQLite DB 연결
  conn = sqlite3.connect("content/" + str(file))

  # if_exists = 'replace' , if_exists = 'append'
  tableDF.to_sql(tableName, conn, if_exists='replace', index=False)

  conn.close()

def AllPlayerPosCos(PtoA, dbName):
    fileName = "playerData" + dbName + ".db"

    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos from ML")
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos from ML")
    # PtoP라면
    if PtoA == 1:
        AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos from ML WHERE ID != " + '"' + dbName + '"')
    elif PtoA == 2:
        PlayerTable = AiTable
        PlayerTable = PlayerTable[PlayerTable['ID'] == dbName]

    saveDBname = int(dbName)

    idCount = A_maxID
    gameCount = A_maxGame

    cosDF = list()
    avgDF = list()

    for Game in range(1, P_maxGame):
        playerFilterTable = PlayerTable[PlayerTable['gameNum'] == Game]
        for ID in range(1, idCount):
            if PtoA == 1 or PtoA == 2:
                if dbName != str(ID):
                    for aiGame in range(1, gameCount):
                        # ai 필터링
                        aiFilter = AiTable[AiTable['ID'] == str(ID)]
                        aiFilter = aiFilter[aiFilter['gameNum'] == aiGame]
                        if len(aiFilter) > 0:
                            playerTablePos = playerFilterTable[['gameNum', 'step', 'xPos', 'yPos']]
                            aiFilterPos = aiFilter[['ID', 'gameNum', 'xPos', 'yPos']]

                            playerTablePos.columns = ['P_Game', 'Step', 'P_xPos', 'P_yPos']
                            aiFilterPos.columns = ['A_ID', 'A_Game', 'A_xPos', 'A_yPos']

                            playerTablePos.reset_index(drop=True, inplace=True)
                            aiFilterPos.reset_index(drop=True, inplace=True)

                            # 두 포지션을 합치고 빈자리를 0으로 만듬
                            contactDF = pd.concat([playerTablePos, aiFilterPos], axis=1)
                            contactDF = contactDF.fillna(0)

                            playerPos = csr_matrix(contactDF[['P_xPos', 'P_yPos']].values)
                            aiPos = csr_matrix(contactDF[['A_xPos', 'A_yPos']].values)
                            # 유사도 계산
                            similarity_simple_pair = cosine_similarity(playerPos, aiPos)
                    
                            tableDF = pd.DataFrame(similarity_simple_pair)
                            tableDF = np.diag(tableDF.values)
                            #avg = np.mean(tableDF)

                            # 유사도 계산
                            #contactDF['Cos'] = cos_sim(contactDF[['P_xPos', 'P_yPos']].values, contactDF[['A_xPos' , 'A_yPos']].values)

                            contactDF['Cos'] = tableDF
                            contactDF['P_ID'] = str(saveDBname)
                            contactDF['P_Game'] = Game
                            contactDF['A_ID'] = str(ID)
                            contactDF['A_Game'] = aiGame

                            cosDF.append(contactDF)

                            avg = np.mean(contactDF['Cos'].values)
                            # 데이터 P - ID, Game, A - ID, Game, Cos
                            appendAvgDF = pd.DataFrame(data=[(str(saveDBname), Game, str(ID), aiGame, avg)],columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                            avgDF.append(appendAvgDF)
            else:
                for aiGame in range(1, gameCount):
                    # ai 필터링
                    aiFilter = AiTable[AiTable['ID'] == str(ID)]
                    aiFilter = aiFilter[aiFilter['gameNum'] == aiGame]
                    if len(aiFilter) > 0:
                        playerTablePos = playerFilterTable[['gameNum', 'step', 'xPos', 'yPos']]
                        aiFilterPos = aiFilter[['ID', 'gameNum', 'xPos', 'yPos']]

                        playerTablePos.columns = ['P_Game', 'Step', 'P_xPos', 'P_yPos']
                        aiFilterPos.columns = ['A_ID', 'A_Game', 'A_xPos', 'A_yPos']

                        playerTablePos.reset_index(drop=True, inplace=True)
                        aiFilterPos.reset_index(drop=True, inplace=True)

                        # 두 포지션을 합치고 빈자리를 0으로 만듬
                        contactDF = pd.concat([playerTablePos, aiFilterPos], axis=1)
                        contactDF = contactDF.fillna(0)

                        playerPos = csr_matrix(contactDF[['P_xPos', 'P_yPos']].values)
                        aiPos = csr_matrix(contactDF[['A_xPos', 'A_yPos']].values)
                        # 유사도 계산
                        similarity_simple_pair = cosine_similarity(playerPos, aiPos)
                    
                        tableDF = pd.DataFrame(similarity_simple_pair)
                        tableDF = np.diag(tableDF.values)
                        #avg = np.mean(tableDF)

                        # 유사도 계산
                        #contactDF['Cos'] = cos_sim(contactDF[['P_xPos', 'P_yPos']].values, contactDF[['A_xPos' , 'A_yPos']].values)

                        contactDF['Cos'] = tableDF
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

    # 플레이어 게임을 기준으로 해서 유사도 분석
    gameAvg = list()
    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for P_Game in range(1, P_maxGame):
        plyerFilter = avgDF[avgDF['P_Game'] == P_Game]
        if len(plyerFilter) > 0:
            for A_ID in range(1, int(checkMaxID)):
                ID = A_ID * A_intervalID
                if PtoA == 1 or PtoA == 2:
                    if dbName != str(ID):
                        aiFilter = plyerFilter[plyerFilter['A_ID'] == str(ID)]
                        aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                        aiFilter = aiFilter.drop_duplicates(['P_Game'])
                        avg = np.mean(aiFilter['Cos'].values)
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
                else:
                    aiFilter = plyerFilter[plyerFilter['A_ID'] == str(ID)]
                    aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                    aiFilter = aiFilter.drop_duplicates(['P_Game'])
                    avg = np.mean(aiFilter['Cos'].values)
                    appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                    gameAvg.append(appendAvgDF)

    avgGameDF = pd.concat(gameAvg)
    avgGameDF = SortByP_GameCos(avgGameDF)
    
    # 여기가 문제인 듯
    avgID_DF = AvgDup(PtoA, dbName, avgDF)
    avgID_DF = avgID_DF.sort_values(by=['Cos'], ascending=False, axis=0)
    
    print('포지션 처리 완료')

    # 모든 작업이 끝나면 저장
    SqlDFSave('posCos' + dbName + '.db', cosDF, 'PosCos')
    SqlDFSave('saveSqlData.db', avgDF, 'PosAvg')
    SqlDFSave('saveSqlData.db', avgGameDF, 'PosGameAvg')
    SqlDFSave('saveSqlData.db', avgID_DF, 'PosIDAvg')

def OverPlayerCos(PtoA, dbName):   
    fileName = "playerData" + dbName + ".db"

    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")

    # PtoP라면
    if PtoA == 1:
        AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0 and ID != " + '"' + dbName + '"')
    elif PtoA == 2:
        PlayerTable = AiTable
        PlayerTable = PlayerTable[PlayerTable['ID'] == dbName]

    P_gameCount = P_maxGame
    idCount = A_maxID
    gameCount = A_maxGame
    saveDBname = int(dbName)

    P_Filter = PlayerTable
    A_Filter = AiTable

    playerTablePos = P_Filter[['gameNum', 'step', 'xPos', 'yPos']]
    aiFilterPos = A_Filter[['ID', 'gameNum', 'step', 'xPos', 'yPos']]

    playerTablePos.columns = ['P_Game', 'P_Step', 'P_xPos', 'P_yPos']
    aiFilterPos.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos']

    # 형변환
    playerOver = csr_matrix(P_Filter[['xPos', 'yPos']].values)
    aiOver = csr_matrix(A_Filter[['xPos', 'yPos']].values)

    # 유사도 계산
    similarity_simple_pair = cosine_similarity(playerOver, aiOver)
                
    tableDF = pd.DataFrame(similarity_simple_pair)

    playerTablePos.reset_index(drop=True, inplace=True)
    aiFilterPos.reset_index(drop=True, inplace=True)

    cosDF = list()

    # 유사도 계산 된 것을 정렬한다.
    for i in range(0, len(tableDF)):
        input = tableDF.iloc[i]
        inDF = pd.concat([aiFilterPos, input], axis=1)
        inDF.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']
        inDF['P_ID'] = str(saveDBname)
        for col in range(0, len(playerTablePos.columns)):
            input = playerTablePos.iat[i, col]
            inDF[playerTablePos.columns[col]] = input
        # 정렬
        inDF.sort_values(by=['Cos'], ascending=False, axis=0)
        cosDF.append(inDF)
    
    cosDF = pd.concat(cosDF)
    cosDF = cosDF[['P_ID', 'P_Game', 'P_Step', 'P_xPos', 'P_yPos', 'A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']]

    gameAvg = list()
    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for P_Game in range(1, P_gameCount):
        plyerFilter = cosDF[cosDF['P_Game'] == P_Game]
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            if PtoA == 1 or PtoA == 2:
                if dbName != str(ID):
                    if len(plyerFilter) > 0:
                        aiFilter = plyerFilter[plyerFilter['A_ID'] == str(ID)]
                        if len(aiFilter) > 0:
                            aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                            aiFilter = aiFilter.drop_duplicates(['P_Game'])
                            avg = np.mean(aiFilter['Cos'].values)
                            appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                            gameAvg.append(appendAvgDF)
                        else:
                            appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                            gameAvg.append(appendAvgDF)
                    else:
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
            else:
                if len(plyerFilter) > 0:
                    aiFilter = plyerFilter[plyerFilter['A_ID'] == str(ID)]
                    if len(aiFilter) > 0:
                        aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                        aiFilter = aiFilter.drop_duplicates(['P_Game'])
                        avg = np.mean(aiFilter['Cos'].values)
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
                    else:
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
                else:
                    appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                    gameAvg.append(appendAvgDF)

    avgGameDF = pd.concat(gameAvg)
    avgGameDF = SortByP_GameCos(avgGameDF)

    avgDF = list()
    for P_Game in range(1, int(P_maxGame)):
        P_Game_Filter = cosDF[cosDF['P_Game'] == P_Game]
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            if PtoA == 1 or PtoA == 2:
                if dbName != A_ID:
                    A_ID_filterDF = P_Game_Filter[P_Game_Filter['A_ID'] == str(ID)]
                    for A_Game in range(1, int(A_maxGame)):
                        A_Game_filterDF = A_ID_filterDF[A_ID_filterDF['A_Game'] == A_Game]
                        if len(A_Game_filterDF) > 0:
                            avg = A_Game_filterDF['Cos'].values[0]
                            appendAvgDF =  pd.DataFrame(data=[(str(dbName), P_Game, str(ID), A_Game, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                            avgDF.append(appendAvgDF)
                        else:
                            appendAvgDF =  pd.DataFrame(data=[(str(dbName), P_Game, str(ID), A_Game, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                            avgDF.append(appendAvgDF)
            else:
                A_ID_filterDF = cosDF[cosDF['A_ID'] == str(ID)]
                for A_Game in range(1, int(A_maxGame)):
                    A_Game_filterDF = A_ID_filterDF[A_ID_filterDF['A_Game'] == A_Game]
                    if len(A_Game_filterDF) > 0:
                        avg = A_Game_filterDF['Cos'].values[0]
                        appendAvgDF =  pd.DataFrame(data=[(str(dbName), P_Game, str(ID), A_Game, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
                    else:
                        appendAvgDF =  pd.DataFrame(data=[(str(dbName), P_Game, str(ID), A_Game, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)

    avgDF = pd.concat(avgDF)
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)
    
    avgID_DF = list()
    # 이제 검사된 값으로 반대로 가장 비슷한 학습량을 찾아냄
    # 평균을 계산할때 Ai와 플레이어의 횟수를 검사한다.
    for A_ID in range(1, int(checkMaxID)):
        ID = A_ID * A_intervalID
        if PtoA == 1 or PtoA == 2:
            if dbName != str(ID):
                # 먼저 필터한다.
                A_filterDF = cosDF[cosDF['A_ID'] == str(ID)]
                # 데이터가 있어야 정리로 넘어감
                if len(A_filterDF) > 0:
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
                    avgID_DF.append(appendAvgDF)
                # 해당하는 아이디에 데이터가 없다면 모두 성공한 것 0으로 데이터를 넣어준다.
                else:
                    #데이터가 없으면 
                    avg = 0
                    appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                    avgID_DF.append(appendAvgDF)
        else:
            # 먼저 필터한다.
            A_filterDF = cosDF[cosDF['A_ID'] == str(ID)]
            # 데이터가 있어야 정리로 넘어감
            if len(A_filterDF) > 0:
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
                avgID_DF.append(appendAvgDF)
            # 해당하는 아이디에 데이터가 없다면 모두 성공한 것 0으로 데이터를 넣어준다.
            else:
                #데이터가 없으면 
                avg = 0
                appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                avgID_DF.append(appendAvgDF)
    
    avgIDDF = pd.concat(avgID_DF)
    avgIDDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('죽음 처리 완료')
    # 모든 작업이 끝나면 DF로 변환 후
    SqlDFSave('overCos' + dbName + '.db', cosDF, 'OverCos')
    SqlDFSave('saveSqlData.db', avgDF, 'OverAvg')
    SqlDFSave('saveSqlData.db', avgGameDF, 'OverGameAvg')
    SqlDFSave('saveSqlData.db', avgIDDF, 'OverIDAvg')

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
                appendDF = filterDF2.iloc[0:1]
                inputList.append(appendDF)
    inDF = pd.concat(inputList)
    return inDF

def JumpFilter(jumpDF, col):
    for i in jumpDF.index:
        if i+1 <= len(jumpDF.index)-1:
            # 인덱스, 컬럼
            value1 = jumpDF.iat[i, col] +1 
            value2 = jumpDF.iat[i+1, col]
            if value1 == value2:
                jumpDF.iat[i, col] = 0
    
    return jumpDF[jumpDF[jumpDF.columns[col]] > 0]

def JumpPlayerCos(PtoA, dbName):
    fileName = "playerData" + dbName + ".db"

    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")

    # PtoP라면
    if PtoA == 1:
        AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0 and ID != " + '"' + dbName + '"')
    elif PtoA == 2:
        PlayerTable = AiTable
        PlayerTable = PlayerTable[PlayerTable['ID'] == dbName]

    P_gameCount = P_maxGame
    idCount = A_maxID
    gameCount = A_maxGame
    saveDBname = int(dbName)

    # 일단 두개다 필터링
    # JumpStep = 5
    P_Filter = JumpFilter(PlayerTable, 5)
    A_Filter = JumpFilter(AiTable, 5)

    playerJump = csr_matrix(P_Filter[['xPos', 'yPos']].values)
    aiJump = csr_matrix(A_Filter[['xPos', 'yPos']].values)
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
    for i in range(0, len(tableDF)):
        input = tableDF.iloc[i]
        inDF = pd.concat([aiFilterPos, input], axis=1)
        inDF.columns = ['A_ID', 'A_Game', 'A_Step', 'A_xPos', 'A_yPos', 'Cos']
        inDF['P_ID'] = saveDBname
        for col in range(0, len(playerTablePos.columns)):
            inDF[playerTablePos.columns[col]] = playerTablePos.iat[i, col]
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
        if len(playerFilter) > 0:
            # 실제 같은 step이 하나도록 수정한다.
            for ID in range(1, idCount):
                if PtoA == 1 or PtoA == 2:
                    if dbName != str(ID):
                        aiFilter = playerFilter[playerFilter['A_ID'] == str(ID)]
                        for aiGame in range(1, gameCount):
                            aiFilter2 = aiFilter[aiFilter['A_Game'] == aiGame]
                            if len(aiFilter2) > 0:
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
                            else:
                                appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), Game, str(ID), aiGame, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                                avgDF.append(appendAvgDF)
                else:
                    aiFilter = playerFilter[playerFilter['A_ID'] == str(ID)]
                    for aiGame in range(1, gameCount):
                        aiFilter2 = aiFilter[aiFilter['A_Game'] == aiGame]
                        if len(aiFilter2) > 0:
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
                        else:
                            appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), Game, str(ID), aiGame, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                            avgDF.append(appendAvgDF)
        else:
            for ID in range(1, idCount):
                for aiGame in range(1, gameCount):
                    appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), Game, str(ID), aiGame, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                    avgDF.append(appendAvgDF)

    avgDF = pd.concat(avgDF)
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    gameAvg = list()
    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for P_Game in range(1, P_gameCount):
        plyerFilter = avgDF[avgDF['P_Game'] == P_Game]
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            if PtoA == 1 or PtoA == 2:
                if dbName != str(ID):
                    if len(plyerFilter) > 0:
                        aiFilter = plyerFilter[plyerFilter['A_ID'] == str(ID)]
                        if len(aiFilter) > 0:
                            aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                            aiFilter = aiFilter.drop_duplicates(['P_Game'])
                            avg = np.mean(aiFilter['Cos'].values)
                            appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                            gameAvg.append(appendAvgDF)
                        else:
                            appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                            gameAvg.append(appendAvgDF)
                    else:
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
            else:
                if len(plyerFilter) > 0:
                    aiFilter = plyerFilter[plyerFilter['A_ID'] == str(ID)]
                    if len(aiFilter) > 0:
                        aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                        aiFilter = aiFilter.drop_duplicates(['P_Game'])
                        avg = np.mean(aiFilter['Cos'].values)
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
                    else:
                        appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                        gameAvg.append(appendAvgDF)
                else:
                    appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                    gameAvg.append(appendAvgDF)

    avgGameDF = pd.concat(gameAvg)
    avgGameDF = SortByP_GameCos(avgGameDF)
    
    """
    avgID_DF = list()
    # 최종
    for A_ID in range(1, int(checkMaxID)):
        ID = A_ID * A_intervalID
        if PtoA == False:
            if dbName != str(ID):
                aiFilter = avgDF[avgDF['A_ID'] == str(ID)]
                aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
                aiFilter = aiFilter.drop_duplicates(['P_Game'])
                avg = np.mean(aiFilter['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                avgID_DF.append(appendAvgDF)
        else:
            aiFilter = avgDF[avgDF['A_ID'] == str(ID)]
            aiFilter = aiFilter.sort_values(by=['Cos'], ascending=False, axis=0)
            aiFilter = aiFilter.drop_duplicates(['P_Game'])
            avg = np.mean(aiFilter['Cos'].values)
            appendAvgDF =  pd.DataFrame(data=[(str(saveDBname), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
            avgID_DF.append(appendAvgDF)
    """
    avgID_DF = AvgDup(PtoA, dbName, avgDF)
    print('점프 처리 완료')

    # 모든 작업이 끝나면 DF로 변환 후
    SqlDFSave('jumpCos' + dbName + '.db', cosDF, 'JumpCos')
    SqlDFSave('saveSqlData.db', avgDF, 'JumpAvg')
    SqlDFSave('saveSqlData.db', avgGameDF, 'JumpGameAvg')
    SqlDFSave('saveSqlData.db', avgID_DF, 'JumpIDAvg')

def TotalGame(PtoA, P_ID):
    OverGameDF = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, Cos from OverGameAvg")
    JumpGameDF = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, Cos from JumpGameAvg")
    CosGameDF = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, Cos from PosGameAvg")

    P_gameCount = P_maxGame
    
    P_OverFilter = OverGameDF[OverGameDF['P_ID'] == str(P_ID)]
    P_JumpFilter = JumpGameDF[JumpGameDF['P_ID'] == str(P_ID)]
    P_CosFilter = CosGameDF[CosGameDF['P_ID'] == str(P_ID)]

    totalGameDF = list()
    checkMaxID = ((A_maxID-1) / A_intervalID) +1

    for P_Game in range(1, P_gameCount):
        P_OverGameFilter = P_OverFilter[P_OverFilter['P_Game'] == P_Game]
        P_JumpGameFilter = P_JumpFilter[P_JumpFilter['P_Game'] == P_Game]
        P_CosGameFilter = P_CosFilter[P_CosFilter['P_Game'] == P_Game]
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            if PtoA == 1 or PtoA == 2:
                if P_ID != str(ID):
                    OverFilter = P_OverGameFilter[P_OverGameFilter['A_ID'] == str(ID)]
                    JumpFilter = P_JumpGameFilter[P_JumpGameFilter['A_ID'] == str(ID)]
                    CosFilter = P_CosGameFilter[P_CosGameFilter['A_ID'] == str(ID)]
                    conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
                    avg = np.mean(conDF['Cos'].values)
                    appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, ID, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                    totalGameDF.append(appendAvgDF)
            else:
                OverFilter = P_OverGameFilter[P_OverGameFilter['A_ID'] == str(ID)]
                JumpFilter = P_JumpGameFilter[P_JumpGameFilter['A_ID'] == str(ID)]
                CosFilter = P_CosGameFilter[P_CosGameFilter['A_ID'] == str(ID)]
                conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
                avg = np.mean(conDF['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, ID, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                totalGameDF.append(appendAvgDF)

    totalGameDF = pd.concat(totalGameDF)
    totalGameDF = SortByP_GameCos(totalGameDF)
    
    print('종합 게임완료')
    SqlDFSave('saveSqlData.db', totalGameDF, 'TotalGameAvg')

def Total(PtoA, P_ID):
    OverDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from OverIDAvg")
    JumpDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from JumpIDAvg")
    CosDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from PosIDAvg")

    totalGameDF = list()

    totalDF = list()

    idCount = A_maxID
    P_OverFilter = OverDF[OverDF['P_ID'] == str(P_ID)]
    P_JumpFilter = JumpDF[JumpDF['P_ID'] == str(P_ID)]
    P_CosFilter = CosDF[CosDF['P_ID'] == str(P_ID)]

    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for A_ID in range(1, int(checkMaxID)):
        ID = A_ID * A_intervalID
        if PtoA == False:
            if P_ID != str(ID):
                OverFilter = P_OverFilter[P_OverFilter['A_ID'] == str(ID)]
                JumpFilter = P_JumpFilter[P_JumpFilter['A_ID'] == ID]
                CosFilter = P_CosFilter[P_CosFilter['A_ID'] == ID]
                conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
                avg = np.mean(conDF['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(str(P_ID), ID, avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                totalDF.append(appendAvgDF)
        else:
            OverFilter = P_OverFilter[P_OverFilter['A_ID'] == str(ID)]
            JumpFilter = P_JumpFilter[P_JumpFilter['A_ID'] == str(ID)]
            CosFilter = P_CosFilter[P_CosFilter['A_ID'] == str(ID)]
            conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
            avg = np.mean(conDF['Cos'].values)
            appendAvgDF =  pd.DataFrame(data=[(str(P_ID), ID, avg)], columns = ['P_ID', 'A_ID', 'Cos'])
            totalDF.append(appendAvgDF)

    totalDF = pd.concat(totalDF)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('종합 완료')
    
    SqlDFSave('saveSqlData.db', totalDF, 'TotalAvg')

def TotalContact():
    totalAvgDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from TotalAvg")

    totalDF = list()

    # #1_P_ID, #1_A_ID, #1_Cos, #2_P_ID, #2_A_ID, #2_Cos, Cos
    for firstP_ID in range(0, len(totalAvgDF)):
        first_P_ID = totalAvgDF['P_ID'][firstP_ID]
        first_A_ID = totalAvgDF['A_ID'][firstP_ID]
        first_Cos = totalAvgDF['Cos'][firstP_ID]
        FilterA_ID = totalAvgDF[totalAvgDF['P_ID'] == str(first_A_ID)]
        FilterA_ID = FilterA_ID[FilterA_ID['A_ID'] == int(first_P_ID)]
        second_Cos = FilterA_ID['Cos'].values[0]
        avg = (first_Cos + second_Cos) / 2
        appendAvgDF =  pd.DataFrame(data=[(first_P_ID, first_A_ID, first_Cos, first_A_ID, first_P_ID, second_Cos, avg)], columns = ['First_P_ID', 'First_A_ID', 'First_Cos', 'Second_P_ID', 'Second_A_ID', 'Second_Cos', 'Cos'])
        totalDF.append(appendAvgDF)

    totalDF = pd.concat(totalDF)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)

    print('종합 완료')
    
    SqlDFSave('saveSqlData.db', totalDF, 'TotalContact')

def AllPlayerData(PtoA, P_ID):
    ### 죽음 데이터
    OverPlayerCos(PtoA, P_ID)

    ### 점프
    JumpPlayerCos(PtoA, P_ID)

    ### 포지션
    AllPlayerPosCos(PtoA, P_ID)

    TotalGame(PtoA, P_ID)

    ### 종합
    Total(PtoA, P_ID)
    print("플레이어 완료: " + P_ID)

def AllPlayersData(PtoA, min, max):
    for player in range(min, max+1):
        AllPlayerData(PtoA, str(player))

def SortByP_GameCos(tableDF):
    sortList = list()
    for game in range(1, P_maxGame):
        filterGame = tableDF[tableDF['P_Game'] == game]
        filterGame['A_ID'] = filterGame['A_ID'].astype(int)
        filterGame = filterGame.sort_values(by=['Cos', 'A_ID'], ascending=[False, True], axis=0)
        filterGame['A_ID'] = filterGame['A_ID'].astype(str)
        sortList.append(filterGame)
    
    sortTable = pd.concat(sortList)
    return sortTable

def PlayerDataCon(min, max):
    playerDataList = list()
    for player in range(min, max+1):
        dbName = 'playerData' + str(player) + '.db'
        PlayerTable = SqlDFLoad(dbName, "select ID, gameNum, step, clearStep, JumpStep, OverStep, xPos, yPos, time from ML")
        PlayerTable['ID'] = str(player)
        playerDataList.append(PlayerTable)
    
    sortTable = pd.concat(playerDataList)
    print('플레이어 데이터 합치기')
    SqlDFSave('sqlSetML.db', sortTable, 'ML')

# 플레이어끼리 비교한 데이터로 사람들끼리 그룹만들기
def PlayersGrouping():
    totalConDF = SqlDFLoad('saveSqlData.db', "select First_P_ID, First_A_ID, First_Cos, Second_P_ID, Second_A_ID, Second_Cos, Cos from TotalContact")

    totalDF = list()

    # 
    for P_ID in range(1, 15):
        FilterP_ID = totalConDF[totalConDF['First_P_ID'] == str(P_ID)]
        first_P_ID = totalAvgDF['P_ID'][firstP_ID]
        first_A_ID = totalAvgDF['A_ID'][firstP_ID]
        first_Cos = totalAvgDF['Cos'][firstP_ID]
        FilterA_ID = totalAvgDF[totalAvgDF['P_ID'] == str(first_A_ID)]
        FilterA_ID = FilterA_ID[FilterA_ID['A_ID'] == int(first_P_ID)]
        second_Cos = FilterA_ID['Cos'].values[0]
        avg = (first_Cos + second_Cos) / 2
        appendAvgDF =  pd.DataFrame(data=[(first_P_ID, first_A_ID, first_Cos, first_A_ID, first_P_ID, second_Cos, avg)], columns = ['First_P_ID', 'First_A_ID', 'First_Cos', 'Second_P_ID', 'Second_A_ID', 'Second_Cos', 'Cos'])
        totalDF.append(appendAvgDF)

    totalDF = pd.concat(totalDF)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)

    print('종합 완료')
    
def AvgDup(PtoA, dbName, avgDF):
    DF_List = list()
    firstDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for A_ID in range(1, int(checkMaxID)):
        ID = A_ID * A_intervalID
        if PtoA == 1 or PtoA == 2:
            if dbName != str(ID):
                FilterTable2 = firstDF[firstDF['A_ID'] == str(ID)]
                aGameList = list()
                pGameList = list()
                if len(FilterTable2) > 0:
                    for P_Game in range(1, P_maxGame):
                        FilterTable3 = FilterTable2
                        for filterP_Game in pGameList:
                            FilterTable3 = FilterTable3[FilterTable3['P_Game'] != filterP_Game]
                        for filterA_Game in aGameList:
                            FilterTable3 = FilterTable3[FilterTable3['A_Game'] != filterA_Game]
                        DF_List.append(FilterTable3.iloc[0:1])
                        pGameList.append(FilterTable3['P_Game'].values[0])
                        aGameList.append(FilterTable3['A_Game'].values[0])
        else:
            FilterTable2 = firstDF[firstDF['A_ID'] == str(ID)]
            aGameList = list()
            pGameList = list()
            if len(FilterTable2) > 0:
                for P_Game in range(1, P_maxGame):
                    FilterTable3 = FilterTable2
                    for filterP_Game in pGameList:
                        FilterTable3 = FilterTable3[FilterTable3['P_Game'] != filterP_Game]
                    for filterA_Game in aGameList:
                        FilterTable3 = FilterTable3[FilterTable3['A_Game'] != filterA_Game]
                    DF_List.append(FilterTable3.iloc[0:1])
                    pGameList.append(FilterTable3['P_Game'].values[0])
                    aGameList.append(FilterTable3['A_Game'].values[0])
    
    conDF = pd.concat(DF_List)

    totalList = list()

    for A_ID in range(1, int(checkMaxID)):
        ID = A_ID * A_intervalID
        if PtoA == False:
            if dbName != str(ID):
                aiFilter = conDF[conDF['A_ID'] == str(ID)]
                avg = np.mean(aiFilter['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(dbName, str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                totalList.append(appendAvgDF)
        else:
            aiFilter = conDF[conDF['A_ID'] == str(ID)]
            avg = np.mean(aiFilter['Cos'].values)
            appendAvgDF =  pd.DataFrame(data=[(dbName, str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
            totalList.append(appendAvgDF)
    
    totalDF = pd.concat(totalList)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)

    return totalDF

def AvgsJump(PtoA, min, max):
    fileName = 'saveSqlData.db'
    
    PosAvgDF = SqlDFLoad(fileName, "select P_ID, P_Game, A_ID, A_Game, Cos from JumpAvg")

    posList = list()

    for P_ID in range(min, max):
        filterDF = PosAvgDF[PosAvgDF['P_ID'] == str(P_ID)]
        PosFilterDF = AvgDup(PtoA, str(P_ID), filterDF)
        print('플레이어 데이터 수정중: ' + str(P_ID))
        posList.append(PosFilterDF)
        
    PosDF = pd.concat(posList)

    print('평균 점프 재설정')
    SqlDFSave('saveSqlData.db', PosDF, 'JumpIDAvg')

def AvgsPos(PtoA, min, max):
    fileName = 'saveSqlData.db'
    
    PosAvgDF = SqlDFLoad(fileName, "select P_ID, P_Game, A_ID, A_Game, Cos from PosAvg")

    posList = list()

    for P_ID in range(min, max):
        filterDF = PosAvgDF[PosAvgDF['P_ID'] == str(P_ID)]
        PosFilterDF = AvgDup(PtoA, str(P_ID), filterDF)
        print('플레이어 데이터 수정중: ' + str(P_ID))
        posList.append(PosFilterDF)
        
    PosDF = pd.concat(posList)

    print('평균 위치 재설정')
    SqlDFSave('saveSqlData.db', PosDF, 'PosIDAvg')

def AvgsReOver():
    fileName = 'saveSqlData.db'
    
    OverAvgDF = SqlDFLoad(fileName, "select P_ID, P_Game, A_ID, A_Game, Cos from OverCos")

    posList = list()

    for P_ID in range(1, P_maxID):
        filterDF = OverAvgDF[OverAvgDF['P_ID'] == str(P_ID)]
        PosFilterDF = AvgDup(False, str(P_ID), filterDF)
        print('플레이어 데이터 수정중: ' + str(P_ID))
        posList.append(PosFilterDF)
        
    PosDF = pd.concat(posList)

    SqlDFSave('saveSqlData.db', PosDF, 'OverAvg')

def AllTotalCal(PtoA, min, max):
    for P_ID in range(min, max):
        Total(PtoA, str(P_ID))
        print('종합 데이터 수정중: ' + str(P_ID))

def TotalGroupA_Check(groupList):
    fileName = 'saveSqlData.db'
    
    avgDF = SqlDFLoad(fileName, "select P_ID, A_ID, Cos from TotalAvg")

    totalList = list()

    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for groupNum in range(0, len(groupList)):
        # 먼저 해당하는 그룹끼리 필터
        tempDFList = list()
        for P_ID in groupList[groupNum]:
            filterDF = avgDF[avgDF['P_ID'] == str(P_ID)]
            tempDFList.append(filterDF)
        tempDF = pd.concat(tempDFList)
        # 이제 A_ID 별로 필터링 한 후 평균을 만든다.
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            filterDF2 = tempDF[tempDF['A_ID'] == ID]
            avg = np.mean(filterDF2['Cos'].values)
            # 그룹, A_ID, Cos
            appendAvgDF =  pd.DataFrame(data=[(str(groupNum + 1), str(ID), avg)], columns = ['GroupName', 'A_ID', 'Cos'])
            totalList.append(appendAvgDF)

    totalDF = pd.concat(totalList)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)

    print('그룹 평균')
    SqlDFSave('saveSqlData.db', totalDF, 'GroupCos')

def TotalGamesCal(min, max):
    OverDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from OverIDCos")
    JumpDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from JumpIDAvg")
    CosDF = SqlDFLoad('saveSqlData.db', "select P_ID, A_ID, Cos from PosIDAvg")

    totalGameDF = list()

    totalDF = list()

    idCount = A_maxID
    checkMaxID = ((A_maxID-1) / A_intervalID) +1

    for P_ID in range(min, max):
        P_OverFilter = OverDF[OverDF['P_ID'] == str(P_ID)]
        P_JumpFilter = JumpDF[JumpDF['P_ID'] == str(P_ID)]
        P_CosFilter = CosDF[CosDF['P_ID'] == str(P_ID)]
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            if PtoA == False:
                if P_ID != str(ID):
                    OverIDFilter = P_OverFilter[P_OverFilter['A_ID'] == str(ID)]
                    JumpIDFilter = P_JumpFilter[P_JumpFilter['A_ID'] == ID]
                    CosIDFilter = P_CosFilter[P_CosFilter['A_ID'] == ID]
                    for game in range(1, A_maxGame):
                        conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)

                        avg = np.mean(conDF['Cos'].values)
                        appendAvgDF =  pd.DataFrame(data=[(str(P_ID), ID, avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                        totalDF.append(appendAvgDF)
            else:
                OverFilter = P_OverFilter[P_OverFilter['A_ID'] == str(ID)]
                JumpFilter = P_JumpFilter[P_JumpFilter['A_ID'] == str(ID)]
                CosFilter = P_CosFilter[P_CosFilter['A_ID'] == str(ID)]
                conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
                avg = np.mean(conDF['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(str(P_ID), ID, avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                totalDF.append(appendAvgDF)

    totalDF = pd.concat(totalDF)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('종합 완료')
    
    SqlDFSave('saveSqlData.db', totalDF, 'TotalAvg')

def PosCosGames():
    PosAvg = SqlDFLoad('aiCosData.db', "select P_ID, P_Game, Step, P_xPos, P_yPos, A_ID, A_Game, A_xPos, A_yPos, Cos from PosCos WHERE P_ID = '1' and P_Game = 1")
    PosAvg.to_csv('content/p1.csv', sep=',', na_rep='NaN')
    PosAvg = SqlDFLoad('aiCosData.db', "select P_ID, P_Game, Step, P_xPos, P_yPos, A_ID, A_Game, A_xPos, A_yPos, Cos from PosCos WHERE P_ID = '14' and P_Game = 10")
    PosAvg.to_csv('content/p2.csv', sep=',', na_rep='NaN')

def OverCheck(PtoA, min, max):
    OverDF = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, P_Step, P_xPos, P_yPos, A_ID, A_Game, A_Step, A_xPos, A_yPos, Cos from OverCos")
    avgDF = list()

    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for P_ID in range(min, max):
        cosDF = OverDF[OverDF['P_ID'] == str(P_ID)]
        for A_ID in range(1, int(checkMaxID)):
            ID = A_ID * A_intervalID
            if PtoA == False:
                if str(P_ID) != str(ID):
                    # 먼저 필터한다.
                    A_filterDF = cosDF[cosDF['A_ID'] == str(ID)]
                    # 데이터가 있어야 정리로 넘어감
                    if len(A_filterDF) > 0:
                        # 유사도 순으로 정렬1
                        A_filterDF = A_filterDF.sort_values(by=['Cos'], ascending=False, axis=0)
                        # 정렬 된 유사도 중 1등만 빼고 삭제한다.
                        A_dupFilterDF = A_filterDF.drop_duplicates(['P_Step', 'P_Game'])
                        A_filterCountDF = A_filterDF.drop_duplicates(['A_Step', 'A_Game'])
                        # 플레이더 데이터가 AI 데이터보다 적을 경우
                        lenP = len(A_dupFilterDF)
                        lenA = len(A_filterCountDF)
                        avg = 0
                        if lenP > lenA:
                            # 정렬을 반대로 바꾼다
                            A_dupFilterDF = A_dupFilterDF.sort_values(by=['Cos'], ascending= True, axis=0)
                            plusNum = lenP - lenA
                            for i in range(0, plusNum):
                                A_dupFilterDF['Cos'].values[i] = 0
                            avg = np.mean(A_filterDF['Cos'].values)
                        elif lenP < lenA:
                            # 그 차만큼 허수를 생성
                            plusNum = lenA - lenP
                            plusArray = np.zeros(plusNum)
                            orginArray = A_dupFilterDF['Cos'].values
                            conNP = np.concatenate((orginArray, plusArray), axis=0)
                            avg = np.mean(conNP)
                        else:
                            avg = np.mean(A_dupFilterDF['Cos'].values)
        
                        appendAvgDF =  pd.DataFrame(data=[(str(P_ID), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                        avgDF.append(appendAvgDF)
                    # 해당하는 아이디에 데이터가 없다면 모두 성공한 것 0으로 데이터를 넣어준다.
                    else:
                        #데이터가 없으면 
                        avg = 0
                        appendAvgDF =  pd.DataFrame(data=[(str(P_ID), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                        avgDF.append(appendAvgDF)
            else:
                # 먼저 필터한다.
                A_filterDF = cosDF[cosDF['A_ID'] == str(ID)]
                # 데이터가 있어야 정리로 넘어감
                if len(A_filterDF) > 0:
                    # 유사도 순으로 정렬1
                    A_filterDF = A_filterDF.sort_values(by=['Cos'], ascending=False, axis=0)
                    # 정렬 된 유사도 중 1등만 빼고 삭제한다.
                    A_dupFilterDF = A_filterDF.drop_duplicates(['P_Step', 'P_Game'])
                    A_filterCountDF = A_filterDF.drop_duplicates(['A_Step', 'A_Game'])
                    # 플레이더 데이터가 AI 데이터보다 적을 경우
                    lenP = len(A_dupFilterDF)
                    lenA = len(A_filterCountDF)
                    avg = 0
                    if lenP > lenA:
                        # 정렬을 반대로 바꾼다
                        A_dupFilterDF = A_filterDF.sort_values(by=['Cos'], ascending= True, axis=0)
                        plusNum = lenP - lenA
                        for i in range(0, plusNum):
                            A_dupFilterDF['Cos'].values[i] = 0
                        avg = np.mean(A_filterDF['Cos'].values)
                    elif lenP < lenA:
                        # 그 차만큼 허수를 생성
                        plusNum = lenA - lenP
                        plusArray = np.zeros(plusNum)
                        orginArray = A_dupFilterDF['Cos'].values
                        conNP = np.concatenate((orginArray, plusArray), axis=0)
                        avg = np.mean(conNP)
                    else:
                        avg = np.mean(A_filterDF['Cos'].values)
        
                    appendAvgDF =  pd.DataFrame(data=[(str(P_ID), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                    avgDF.append(appendAvgDF)
                # 해당하는 아이디에 데이터가 없다면 모두 성공한 것 0으로 데이터를 넣어준다.
                else:
                    #데이터가 없으면 
                    avg = 0
                    appendAvgDF =  pd.DataFrame(data=[(str(P_ID), str(ID), avg)], columns = ['P_ID', 'A_ID', 'Cos'])
                    avgDF.append(appendAvgDF)
        print('죽음 완료 : ' + str(P_ID))

    avgDF = pd.concat(avgDF)
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)
    SqlDFSave('saveSqlData.db', avgDF, 'OverIDAvg')

def OverGame(PtoA):
    OverDF = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, A_Game, Cos from OverCos")

    print('게임 오버 시작')
    avgDF = list()
    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for P_ID in range(1, int(P_maxID)):
        P_ID_Filter = OverDF[OverDF['P_ID'] == str(P_ID)]
        for P_Game in range(1, int(P_maxGame)):
            P_Game_Filter = P_ID_Filter[P_ID_Filter['P_Game'] == P_Game]
            for A_ID in range(1, int(checkMaxID)):
                ID = A_ID * A_intervalID
                if PtoA == False:
                    if P_ID != A_ID:
                        A_ID_filterDF = P_Game_Filter[P_Game_Filter['A_ID'] == str(ID)]
                        for A_Game in range(1, int(A_maxGame)):
                            A_Game_filterDF = A_ID_filterDF[A_ID_filterDF['A_Game'] == A_Game]
                            if len(A_Game_filterDF) > 0:
                                avg = A_Game_filterDF['Cos'].values[0]
                                appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, str(ID), A_Game, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                                avgDF.append(appendAvgDF)
                            else:
                                appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, str(ID), A_Game, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                                avgDF.append(appendAvgDF)
                else:
                    A_ID_filterDF = P_Game_Filter[P_Game_Filter['A_ID'] == str(ID)]
                    for A_Game in range(1, int(A_maxGame)):
                        A_Game_filterDF = A_ID_filterDF[A_ID_filterDF['A_Game'] == A_Game]
                        if len(A_Game_filterDF) > 0:
                            avg = A_Game_filterDF['Cos'].values[0]
                            appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, str(ID), A_Game, avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                            avgDF.append(appendAvgDF)
                        else:
                            appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, str(ID), A_Game, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                            avgDF.append(appendAvgDF)

        print('플레이어 완료: ' + str(P_ID))

    avgDF = pd.concat(avgDF)
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)
    SqlDFSave('saveSqlData.db', avgDF, 'OverAvg')

def HitFilter(tableDF):
    tableDF = tableDF['A_ID'].astype(int)
    tableDF = tableDF['P_ID'].astype(int)
    convertDF =  sorted(tableDF, key = lambda x : x[0])
    sortList = list()
    for game in range(1, P_maxGame):
        filterGame = tableDF[tableDF['P_Game'] == game]
        filterGame['A_ID'] = filterGame['A_ID'].astype(int)
        filterGame = filterGame.sort_values(by=['Cos', 'A_ID'], ascending=[False, True], axis=0)
        filterGame['A_ID'] = filterGame['A_ID'].astype(str)
        sortList.append(filterGame)
    
    sortTable = pd.concat(sortList)
    return sortTable

def OverIDGame(PtoA):
    OverDF = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, A_Game, Cos from OverAvg")

    avgDF = list()
    # 이제 검사된 값으로 반대로 가장 비슷한 학습량을 찾아냄
    # 평균을 계산할때 Ai와 플레이어의 횟수를 검사한다.
    for P_ID in range(1, int(P_maxID)):
        P_ID_filter = OverDF[OverDF['P_ID'] == str(P_ID)]
        avgDF.append(AvgDup(PtoA, P_ID, P_ID_filter))    
        print('플레이어 오버: ' + str(P_ID))

    avgDF = pd.concat(avgDF)
    #avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)
    print('죽음 처리 완료')
    SqlDFSave('saveSqlData.db', avgDF, 'OverIDAvg')

def FilterPoscos():
    for P_ID in range(11, 15):
        ppDF = SqlDFLoad('aiCosData.db', 'select P_ID, P_Game, Step, P_xPos, P_yPos, A_ID, A_Game, A_xPos, A_yPos, Cos from PosCos WHERE P_ID = \"' + str(P_ID) + '\"')
        SqlDFSave('Poscos' + str(P_ID) + '.db' , ppDF, 'PosCos')
        print('필터: ' + str(P_ID))

def AtoA():
    checkMaxID = ((A_maxID-1) / A_intervalID) +1
    for P_ID in range(1, int(checkMaxID)):
        ID = P_ID * A_intervalID
        AllPlayerData(2, str(ID))

#TotalGroupA_Check([(2,6), (8,4,13), (5,11)])
#AllPlayersData(2, 1, 121)
#AtoA()

### 포지션
AllPlayerPosCos(2, '4')