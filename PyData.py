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

def SqlAtoADFLoad(file, execute):
  # SQLite DB 연결
  conn = sqlite3.connect("D:/GoogleDrive/강화학습DB백업/AtoA/" + str(file))

  df = pd.read_sql(execute, conn)

  conn.close()

  return df

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
    PlayerTable = []
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos from ML")

    # PtoP라면
    if PtoA == 0:
        PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos from ML")
    elif PtoA == 1:
        PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos from ML")
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
    PlayerTable = []

    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")

    # PtoP라면
    if PtoA == 0:
        PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")
    elif PtoA == 1:
        PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE overStep > 0")
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
    
    cosDF = list()
    avgGameDF = list()
    avgDF = list()
    avgIDDF = list()

    if len(playerTablePos) > 0:
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
                        A_filterDF = A_filterDF.drop_duplicates(['P_Game', 'P_Step'])
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
                    A_filterDF = A_filterDF.drop_duplicates(['P_Game', 'P_Step'])
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
    else:
        gameAvg = list()
        avgDF = list()
        avgID_DF = list()

        checkMaxID = ((A_maxID-1) / A_intervalID) +1
        for P_Game in range(1, P_gameCount):
            for A_ID in range(1, int(checkMaxID)):
                ID = A_ID * A_intervalID
                for A_Game in range(1, A_maxGame):
                    appendAvgDF =  pd.DataFrame(data=[(str(dbName), P_Game, str(ID), 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                    gameAvg.append(appendAvgDF)
                    appendAvgDF =  pd.DataFrame(data=[(str(dbName), P_Game, str(ID), A_Game, 0)], columns = ['P_ID', 'P_Game', 'A_ID', 'A_Game', 'Cos'])
                    avgDF.append(appendAvgDF)
                    appendAvgDF =  pd.DataFrame(data=[(str(dbName), str(ID), 0)], columns = ['P_ID', 'A_ID', 'Cos'])
                    avgID_DF.append(appendAvgDF)

        avgGameDF = pd.concat(gameAvg)
        avgDF = pd.concat(avgDF)
        avgIDDF = pd.concat(avgID_DF)

    print('죽음 처리 완료')
    # 모든 작업이 끝나면 DF로 변환 후
    if len(playerTablePos) > 0:
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
        count = len(jumpDF.index)-1
        if i+1 <= count:
            # 인덱스, 컬럼
            value1 = jumpDF.iat[i, col] +1 
            value2 = jumpDF.iat[i+1, col]
            if value1 == value2:
                jumpDF.iat[i, col] = 0
    
    return jumpDF[jumpDF[jumpDF.columns[col]] > 0]

def JumpPlayerCos(PtoA, dbName):
    fileName = "playerData" + dbName + ".db"

    # A의 자료를 B의 자료에서 확인하여 가장 높은 평균 유사도의 게임을 찾아낸다.
    PlayerTable = []
    
    AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")

    # PtoP라면
    if PtoA == 0:
        PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")
    elif PtoA == 1:
        PlayerTable = SqlDFLoad(fileName, "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0")
        AiTable = SqlDFLoad("sqlSetML.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE JumpStep > 0 and ID != " + '"' + dbName + '"')
    elif PtoA == 2:
        PlayerTable = AiTable
        PlayerTable = PlayerTable[PlayerTable['ID'] == str(dbName)]

    P_gameCount = P_maxGame
    idCount = A_maxID
    gameCount = A_maxGame
    saveDBname = int(dbName)

    PlayerTable.reset_index(drop=True, inplace=True)
    AiTable.reset_index(drop=True, inplace=True)

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
        ID = P_ID * A_intervalID
        Total(PtoA, str(ID))
        print('종합 데이터 수정중: ' + str(ID))

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
    for P_ID in range(1, 61):
        ID = P_ID * 2
        P_ID_filter = OverDF[OverDF['P_ID'] == str(ID)]
        avgDF.append(AvgDup(PtoA, str(ID), P_ID_filter))    
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

def AtoA(min, max):
    min = min / A_intervalID
    max = max / A_intervalID
    for P_ID in range(int(min), int(max)+1):
        ID = P_ID * A_intervalID
        AllPlayerData(2, str(ID))

def PosSelfSimilarity(PtoA, ID):
    table = SqlDFLoad("aiDatas.db", "select ID, gameNum, step, xPos, yPos from ML WHERE ID == '" + ID + "'")
    if PtoA == 1:
        table = SqlDFLoad("playerDatas.db", "select ID, gameNum, step, xPos, yPos from ML WHERE ID == '" + ID +"'")
    
    IDdf = table
    avgDF = list()
    cosDF = list()
    for P_Game in range(1, P_maxGame):
        a_gameDF = IDdf[IDdf['gameNum'] == P_Game]
        if len(a_gameDF) > 0:
            for A_Game in range(1, P_maxGame):
                if P_Game != A_Game:
                    b_gameDF = IDdf[IDdf['gameNum'] == A_Game]
                    if len(b_gameDF) > 0:
                        playerTablePos = a_gameDF[['gameNum', 'step', 'xPos', 'yPos']]
                        aiFilterPos = b_gameDF[['gameNum', 'xPos', 'yPos']]

                        playerTablePos.columns = ['P_Game', 'Step', 'P_xPos', 'P_yPos']
                        aiFilterPos.columns = ['A_Game', 'A_xPos', 'A_yPos']

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

                        contactDF['Cos'] = tableDF
                        contactDF['P_ID'] = str(ID)
                        contactDF['P_Game'] = P_Game
                        contactDF['A_Game'] = A_Game

                        cosDF.append(contactDF)

                        avg = np.mean(contactDF['Cos'].values)
                        # 데이터 P - ID, Game, A - ID, Game, Cos
                        appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, avg)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
                    else:
                        appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, 0)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
        else:
            for A_Game in range(1, P_maxGame):
                appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, 0)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                avgDF.append(appendAvgDF)
    
    conDF = pd.concat(cosDF)
    avgDF = pd.concat(avgDF)
    # 정렬
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    avg = np.mean(avgDF['Cos'].values)
    avgIDDF = pd.DataFrame(data=[(str(ID), avg)],columns = ['ID','Cos'])
    
    dbName = "Ai_"
    if PtoA == 1:
        dbName = "Human_"
    
    print('위치 완료')
    SqlDFSave(dbName + 'SS' + ID + '.db', conDF, 'PosCos')
    SqlDFSave(dbName + 'SS.db', avgDF, 'PosAvg')
    SqlDFSave(dbName + 'SS.db', avgIDDF, 'PosIDAvg')

def JumpSelfSimilarity(PtoA, ID):
    table = SqlDFLoad("aiDatas.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE ID == " + ID + " and JumpStep > 0")
    if PtoA == 1:
        table = SqlDFLoad("playerDatas.db", "select ID, gameNum, step, xPos, yPos, JumpStep from ML WHERE ID == " + ID + " and JumpStep > 0")

    IDdf = JumpFilter(table, 5)

    avgDF = list()
    cosDF = list()
    for P_Game in range(1, P_maxGame):
        a_gameDF = IDdf[IDdf['gameNum'] == P_Game]
        if len(a_gameDF) > 0:
            for A_Game in range(1, P_maxGame):
                if P_Game != A_Game:
                    b_gameDF = IDdf[IDdf['gameNum'] == A_Game]
                    if len(b_gameDF) > 0:
                        playerTablePos = a_gameDF[['gameNum', 'step', 'xPos', 'yPos']]
                        aiFilterPos = b_gameDF[['gameNum', 'xPos', 'yPos']]

                        playerTablePos.columns = ['P_Game', 'Step', 'P_xPos', 'P_yPos']
                        aiFilterPos.columns = ['A_Game', 'A_xPos', 'A_yPos']

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

                        contactDF['Cos'] = tableDF
                        contactDF['P_ID'] = str(ID)
                        contactDF['P_Game'] = P_Game
                        contactDF['A_Game'] = A_Game

                        cosDF.append(contactDF)

                        avg = np.mean(contactDF['Cos'].values)
                        # 데이터 P - ID, Game, A - ID, Game, Cos
                        appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, avg)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
                    else:
                        appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, 0)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
        else:
            for A_Game in range(1, P_maxGame):
                appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, 0)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                avgDF.append(appendAvgDF)
    
    conDF = pd.concat(cosDF)
    avgDF = pd.concat(avgDF)
    # 정렬
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    avg = np.mean(avgDF['Cos'].values)
    avgIDDF = pd.DataFrame(data=[(str(ID), avg)],columns = ['ID','Cos'])
    
    dbName = "Ai_"
    if PtoA == 1:
        dbName = "Human_"
    
    print('점프 완료')
    SqlDFSave(dbName + 'SS' + ID + '.db', conDF, 'JumpCos')
    SqlDFSave(dbName + 'SS.db', avgDF, 'JumpAvg')
    SqlDFSave(dbName + 'SS.db', avgIDDF, 'JumpIDAvg')

def OverSelfSimilarity(PtoA, ID):
    table = SqlDFLoad("aiDatas.db", "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE ID == " + ID + " and overStep > 0")
    if PtoA == 1:
        table = SqlDFLoad("playerDatas.db", "select ID, gameNum, step, xPos, yPos, overStep from ML WHERE ID == " + ID + " and overStep > 0")

    IDdf = table
    avgDF = list()
    cosDF = list()
    for P_Game in range(1, P_maxGame):
        a_gameDF = IDdf[IDdf['gameNum'] == P_Game]
        if len(a_gameDF) > 0:
            for A_Game in range(1, P_maxGame):
                if P_Game != A_Game:
                    b_gameDF = IDdf[IDdf['gameNum'] == A_Game]
                    if len(b_gameDF) > 0:
                        playerTablePos = a_gameDF[['gameNum', 'step', 'xPos', 'yPos']]
                        aiFilterPos = b_gameDF[['gameNum', 'xPos', 'yPos']]

                        playerTablePos.columns = ['P_Game', 'Step', 'P_xPos', 'P_yPos']
                        aiFilterPos.columns = ['A_Game', 'A_xPos', 'A_yPos']

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

                        contactDF['Cos'] = tableDF
                        contactDF['P_ID'] = str(ID)
                        contactDF['P_Game'] = P_Game
                        contactDF['A_Game'] = A_Game

                        cosDF.append(contactDF)

                        avg = np.mean(contactDF['Cos'].values)
                        # 데이터 P - ID, Game, A - ID, Game, Cos
                        appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, avg)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
                    else:
                        appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, 0)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                        avgDF.append(appendAvgDF)
        else:
            for A_Game in range(1, P_maxGame):
                appendAvgDF = pd.DataFrame(data=[(str(ID), P_Game, A_Game, 0)],columns = ['P_ID', 'P_Game', 'A_Game', 'Cos'])
                avgDF.append(appendAvgDF)
    
    avgDF = pd.concat(avgDF)
    # 정렬
    avgDF = avgDF.sort_values(by=['Cos'], ascending=False, axis=0)

    avg = np.mean(avgDF['Cos'].values)
    avgIDDF = pd.DataFrame(data=[(str(ID), avg)],columns = ['ID','Cos'])
    
    dbName = "Ai_"
    if PtoA == 1:
        dbName = "Human_"
    
    print('죽음 완료')
    if len(cosDF) > 0:
        conDF = pd.concat(cosDF)
        SqlDFSave(dbName + 'SS' + ID + '.db', conDF, 'OverCos')
    SqlDFSave(dbName + 'SS.db', avgDF, 'OverAvg')
    SqlDFSave(dbName + 'SS.db', avgIDDF, 'OverIDAvg')

def TotalSelfSimilarity(PtoA, ID):
    
    dbName = "Ai_"
    if PtoA == 1:
        dbName = "Human_"
    
    overDF = SqlDFLoad(dbName +'SS.db', "select P_ID, P_Game, A_Game, Cos from OverAvg WHERE P_ID == " + ID)
    jumpDF = SqlDFLoad(dbName + 'SS.db', "select P_ID, P_Game, A_Game, Cos from JumpAvg WHERE P_ID == " + ID)
    posDF = SqlDFLoad(dbName + 'SS.db', "select P_ID, P_Game, A_Game, Cos from PosAvg WHERE P_ID == " + ID)

    totalGameDF = list()

    totalDF = list()

    for P_Game in range(1, A_maxGame):
        OverFilter = overDF[overDF['P_Game'] == P_Game]
        JumpFilter = jumpDF[jumpDF['P_Game'] == P_Game]
        CosFilter = posDF[posDF['P_Game'] == P_Game]
        for A_Game in range(1, A_maxGame):
            if P_Game != A_Game:
                A_OverFilter = OverFilter[OverFilter['A_Game'] == A_Game]
                A_JumpFilter = JumpFilter[JumpFilter['A_Game'] == A_Game]
                A_CosFilter = CosFilter[CosFilter['A_Game'] == A_Game]
                conDF = pd.concat([OverFilter, JumpFilter, CosFilter], axis = 0)
                avg = np.mean(conDF['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(str(ID), P_Game, A_Game, avg)], columns = ['P_ID', 'P_Game', 'A_Gmae', 'Cos'])
                totalDF.append(appendAvgDF)

    totalDF = pd.concat(totalDF)
    totalDF = totalDF.sort_values(by=['Cos'], ascending=False, axis=0)

    avg = np.mean(conDF['Cos'].values)
    totalIDDF =  pd.DataFrame(data=[(str(ID), P_Game, A_Game, avg)], columns = ['P_ID', 'P_Game', 'A_Gmae', 'Cos'])

    print('종합 완료')
    
    SqlDFSave(dbName +'SS.db', totalDF, 'TotalAvg')
    SqlDFSave(dbName +'SS.db', totalIDDF, 'TotalIDAvg')

def AllSS(PtoA, min, max):
    A_intervalID = 1
    if PtoA == 0:
        A_intervalID = 2
        min = min / A_intervalID
        max = max / A_intervalID

    for A_ID in range(int(min), int(max)+1):
        ID = A_ID * A_intervalID
        PosSelfSimilarity(PtoA, str(ID))
        JumpSelfSimilarity(PtoA, str(ID))
        OverSelfSimilarity(PtoA, str(ID))
        TotalSelfSimilarity(PtoA, str(ID))
        print('자기유사도 완료: ' + str(ID))

def LenAtoA():
    totalDF = []
    decList = ['posCos', 'jumpCos', 'overCos']
    for dec in decList:
        count = 0
        for i in range(1, 61):
            ID = i*2
            sql = SqlAtoADFLoad(dec + str(ID) +'.db', 'select P_ID from ' + dec)
            count += len(sql)
            print('카운트' + str(ID) + ' ' + str(count))
        appendAvgDF =  pd.DataFrame(data=[(dec, count)], columns = ['Type', 'Count'])
        totalDF.append(appendAvgDF)
        print('완료' + dec + ' ' + str(count))

    totalDF = pd.concat(totalDF)
    SqlDFSave('len.db', totalDF, 'TotalAvg')

def GroupS(groupList, fromDec):
    sqlTable = SqlDFLoad('playersCosData.db', "select P_ID, P_Game, A_ID, A_Game, Cos from " + fromDec)
    DF_List = list()
    for group in groupList:
        # 기준이 되는 첫번째
        zeroPlayer = group[0]
        filterSql = sqlTable[sqlTable['P_ID'] == str(zeroPlayer)]
        for p in range(1, len(group)):
            FilterTable2 = filterSql[filterSql['A_ID'] == str(group[p])]
            #중복 되지 않도록해서 넣는다.
            aGameList = list()
            for P_Game in range(1, 11):
                FilterTable3 = FilterTable2[FilterTable2['P_Game'] == P_Game]
                FilterTable3 = FilterTable3.sort_values(by=['Cos'], ascending=False, axis=0)
                # 필터
                for filterA_Game in aGameList:
                    FilterTable3 = FilterTable3[FilterTable3['A_Game'] != filterA_Game]
                DF_List.append(FilterTable3.iloc[0:1])
                aGameList.append(FilterTable3['A_Game'].values[0])

    conDF = pd.concat(DF_List)
    SqlDFSave('GameAvg.db', conDF, fromDec)

def AllGroupS(groupList):
    avglist = ['PosAvg', 'OverAvg', 'JumpAvg', 'TotalByGameAvg']
    for avg in avglist:
        GroupS(groupList, avg)

def AllGroupAIGame(gList):
    avglist = ['OverAvg', 'PosAvg', 'JumpAvg', 'TotalByGameAvg']
    for avg in range(0, len(avglist)):
        GroupAIS(gList, avglist[avg])

def GroupAIS(groupList, fromDec):
    sqlTable = SqlDFLoad('GameAvg.db', "select P_ID, P_Game, A_ID, A_Game, Cos from TotalByGameAvg")
    sqlAITable = SqlDFLoad('GameByAvg.db', "select P_ID, P_Game, A_ID, Cos from " + fromDec)
    DF_List = list()
    groupID = 1
    for group in groupList:
        # 기준이 되는 첫번째
        zeroPlayer = group[0]
        filterSql = sqlTable[sqlTable['P_ID'] == str(zeroPlayer)]
        meanList = list()
        for game in range(1, 11):
            filterGame = filterSql[filterSql['P_Game'] == game]
            # 기본
            FilterTableAI1 = sqlAITable[sqlAITable['P_ID'] == str(zeroPlayer)]
            FilterTableAI1 = FilterTableAI1[FilterTableAI1['P_Game'] == game]
            FilterTableAI1 = FilterTableAI1.astype({'A_ID': 'int'})
            FilterTableAI1 = FilterTableAI1.sort_values(by=['A_ID'], ascending=True, axis=0)
            FilterTableAI1.reset_index(drop=True, inplace=True)
            
            conDF = FilterTableAI1['A_ID']
            meanList.append(FilterTableAI1['Cos'])
            #print(FilterTableAI1)

            for p in range(0, len(filterGame)):
                g_Player = filterGame['A_ID'].values[p]
                g_Game = filterGame['A_Game'].values[p]

                FilterTableAI2 = sqlAITable[sqlAITable['P_ID'] == str(g_Player)]
                FilterTableAI2 = FilterTableAI2[FilterTableAI2['P_Game'] == g_Game]
                FilterTableAI2 = FilterTableAI2.astype({'A_ID': 'int'})
                FilterTableAI2 = FilterTableAI2.sort_values(by=['A_ID'], ascending=True, axis=0)
                FilterTableAI2.reset_index(drop=True, inplace=True)
                
                #print(FilterTableAI2)
                meanList.append(FilterTableAI2['Cos'])

            #
            contactDF = pd.concat(meanList, axis=1)
            meanDF = contactDF.mean(axis=1)
            conDF = pd.concat([conDF, meanDF], axis=1)
            conDF.columns = ['A_ID', 'Cos']
            conDF['G_ID'] = str(groupID)
            conDF['G_Game'] = game
            conDF = conDF[['G_ID','G_Game','A_ID','Cos']]
            conDF = conDF.sort_values(by=['Cos'], ascending=False, axis=0)
            DF_List.append(conDF)
            #print(conDF)
        groupID += 1
    
    conDF = pd.concat(DF_List)
    SqlDFSave('GameAvgTotal.db', conDF, fromDec)

def TotalAidu():
    sqlAITable = SqlDFLoad('aiCosData.db', "select P_ID, P_Game, A_ID, Cos from TotalGameAvg")
    sqlAITable = sqlAITable.drop_duplicates(['P_ID', 'P_Game', 'A_ID'])
    SqlDFSave('aiCosData.db', sqlAITable, 'TotalGameAvg')


def AllReGameAvg():
    avglist = ['PosAvg', 'OverAvg', 'JumpAvg', 'TotalByGameAvg']
    for avg in range(0, len(avglist)):
        ReGameAvg(avglist[avg])

def ReGameAvg(fromDec):
    sqlAITable = SqlDFLoad('aiCosData.db', "select P_ID, P_Game, A_ID, A_Game, Cos from " + fromDec)
    gameAvg = list()
    for P_ID in range(1, 15):
        p_filter = sqlAITable[sqlAITable['P_ID'] == str(P_ID)]
        for P_Game in range(1, 11):
            pg_filter = p_filter[p_filter['P_Game'] == P_Game]        
            for A_ID in range(1, 61):
                ID = A_ID *2
                a_filter = pg_filter[pg_filter['A_ID'] == str(ID)]
                avg = 0
                if len(a_filter) > 0:
                    avg = np.mean(a_filter['Cos'].values)
                appendAvgDF =  pd.DataFrame(data=[(str(P_ID), P_Game, str(ID), avg)], columns = ['P_ID', 'P_Game', 'A_ID', 'Cos'])
                gameAvg.append(appendAvgDF)

    avgGameDF = pd.concat(gameAvg)
    #avgGameDF = SortByP_GameCos(avgGameDF)
    print('완료' + fromDec)
    SqlDFSave('GameByAvg.db', avgGameDF, str(fromDec))

def ReTotalAvg():
    sqlAITablePos = SqlDFLoad('GameByAvg.db', "select P_ID, P_Game, A_ID, Cos from PosAvg")
    sqlAITableOver = SqlDFLoad('GameByAvg.db', "select P_ID, P_Game, A_ID, Cos from OverAvg")
    sqlAITableJump = SqlDFLoad('GameByAvg.db', "select P_ID, P_Game, A_ID, Cos from JumpAvg")
    
    baseTable = sqlAITablePos[['P_ID', 'P_Game', 'A_ID']]
    avgGameDF = pd.concat([sqlAITablePos['Cos'], sqlAITableOver['Cos'], sqlAITableJump['Cos']], axis = 1)
    meanDF = avgGameDF.mean(axis=1)
    conDF = pd.concat([baseTable, meanDF], axis=1)
    conDF.columns = ['P_ID', 'P_Game', 'A_ID', 'Cos']

    SqlDFSave('GameByAvg.db', conDF, 'TotalByGameAvg')

def ReTotalAvg():
    sqlAITablePos = SqlDFLoad('GroupGameAvgTotal.db', "select G_ID, G_Game, A_ID, Cos from PosAvg")
    sqlAITableOver = SqlDFLoad('GroupGameAvgTotal.db', "select G_ID, G_Game, A_ID, Cos from OverAvg")
    sqlAITableJump = SqlDFLoad('GroupGameAvgTotal.db', "select G_ID, G_Game, A_ID, Cos from JumpAvg")
    sqlAITablePos = sqlAITablePos.astype({'A_ID': 'int'})
    sqlAITableOver = sqlAITablePos.astype({'A_ID': 'int'})
    sqlAITableJump = sqlAITablePos.astype({'A_ID': 'int'})
    sqlAITablePos = sqlAITablePos.sort_values(by=['G_ID', 'G_Game', 'A_ID'], ascending=True)
    sqlAITableOver =sqlAITableOver.sort_values(by=['G_ID', 'G_Game', 'A_ID'], ascending=True)
    sqlAITableJump=  sqlAITableJump.sort_values(by=['G_ID', 'G_Game', 'A_ID'], ascending=True)

    baseTable = sqlAITablePos[['G_ID', 'G_Game', 'A_ID']]
    avgGameDF = pd.concat([sqlAITablePos['Cos'], sqlAITableOver['Cos'], sqlAITableJump['Cos']], axis = 1)
    meanDF = avgGameDF.mean(axis=1)
    conDF = pd.concat([baseTable, meanDF], axis=1)
    conDF.columns = ['G_ID', 'G_Game', 'A_ID', 'Cos']

    SqlDFSave('GroupGameAvgTotal.db', conDF, 'TotalByGameAvg')

def LenSS(A):
    fromList = ['OverCos','JumpCos', 'PosCos']
    totalDF = list()
    if A == True:
        for fromDec in fromList:
            lens = 0
            for aid in range(1, 61):
                id = aid *2
                if fromDec != 'OverCos':
                    sqlTable = SqlDFLoad('Ai_SS' + str(id) + '.db', "select P_ID from " + fromDec)
                    lens += len(sqlTable)
                else:
                    if id != 112 and id != 114:
                        sqlTable = SqlDFLoad('Ai_SS' + str(id) + '.db', "select P_ID from " + fromDec)
                        lens += len(sqlTable)
                print('확인' + str(id))
            appendAvgDF =  pd.DataFrame(data=[(fromDec, lens)], columns = ['Type', 'Count'])
            SqlDFSave('lenSS.db', appendAvgDF, 'AI' + fromDec)
    else:
        for fromDec in fromList:
            lens = 0
            for id in range(1, 15):
                sqlTable = SqlDFLoad('Human_SS' + str(id) + '.db', "select P_ID from " + fromDec)
                lens += len(sqlTable)

            appendAvgDF =  pd.DataFrame(data=[(fromDec, lens)], columns = ['Type', 'Count'])
            SqlDFSave('lenSS.db', appendAvgDF, 'Human' + fromDec)

def ReJumpAtoA():
    for aid in range(1, 61):
        ID = aid *2
        JumpPlayerCos(2, str(ID))
        print('점프')

def ReTotal():
    sqlAITableAvg = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, Cos from JumpAvg")
    sqlAITableGameAvg = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, Cos from JumpGameAvg")
    sqlAITableIDAvg = SqlDFLoad('saveSqlData.db', "select P_ID, P_Game, A_ID, Cos from JumpIDAvg")
    SqlDFSave('AtoA.db', sqlAITableAvg, 'JumpAvg')
    SqlDFSave('AtoA.db', sqlAITableGameAvg, 'JumpGameAvg')
    SqlDFSave('AtoA.db', sqlAITableIDAvg, 'JumpIDAvg')

#TotalGroupA_Check([(2,6), (8,4,13), (5,11)])
#AllPlayersData(2, 1, 121)
#AtoA(112, 120)

### 포지션
#AllPlayerPosCos(2, '4')

#OverPlayerCos(2, '112')

#OverIDGame(2)

#AllTotalCal(2, 2, 61)

#TotalContact()

#LenAtoA()

#AllGroupS([(2,6), (4,8,13), (5,11)])
#AllGroupAIGame([(2,6), (4,8,13), (5,11)])

#TotalAidu()

#AllReGameAvg()

#LenSS(False)

#ReTotalAvg()

ReTotal()

for aid in range(1,61):
    ID = aid *2
    Total(2, str(ID))