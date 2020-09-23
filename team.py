import requests

# access_token='eyJ0eXAiOiAiVENWMiJ9.RHR0MlVVMFVXcDVRcngzY21qYXh6TWtIemM4.MjJjNWJiMmQtMDRkYS00MDg0LTk3ZGEtNmNmYzBlNTc0YTFj'
access_token='eyJ0eXAiOiAiVENWMiJ9.Q0hLRHFCYkQ0UWJpRjJqUDRKX1JIU3NzXzN3.Y2E4M2VlZTMtNGY2My00YmI4LTg4NTUtMTBlMzA5ZTg4ZTA2'

# URL to get changes to buildqueue list
# url  = 'http://teamcity:8111/app/rest/buildQueue?locator=buildType:(id:RdxCode_GerritPendingCi)'
#
# # Url to get buildqueue specific if
# # url='http://teamcity:8111/app/rest/buildQueue/id:375514'
#
# # url to get changes of specific build
# #url='http://teamcity:8111/app/rest/changes?locator=build:(id:375511)'
#
# #url
# = 'http://teamcity:8111/app/rest/problemOccurrences/problem:(id:137),build:(id:373501)'
#
# # url = 'http://teamcity:8111/app/rest/problems/id:137'
#
# # url = 'http://teamcity:8111/app/rest/vcs-roots/id:RdxCode_GerritPendingChanges'
# # url='http://teamcity:8111/app/rest/changes?locator=buildType:(id:RdxCode_GerritPendingCi),pending:true'

url = 'http://localhost:8111/app/rest/buildQueue'
headers = {'Authorization' : 'Bearer ' + access_token}
response = requests.get(url, headers=headers)

print(response.text)
