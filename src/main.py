import pygame
import random

from beat import get_beat_list




#config
audio_file = '../Alex MakeMusic - I Am Good - Instrumental Version.mp3'

#获取音乐beat
beat_frame = get_beat_list(audio_file)

# 初始化pygame
pygame.init()

# 设置窗口大小和标题
win_width, win_height = 800, 600
win_title = "Ball Collision"
screen = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption(win_title)

# 设置地图
map_width, map_height = 2000, 2000
map_color = (0, 0, 255)
map_pos = [0, 0]

# 设置小球
ball_radius = 20
ball_color = (255, 0, 0)
ball_pos = [400, 300]
ball_speed = [5, 5]




# 帧计数器
frame_count = 0

# 加载音乐
pygame.mixer.music.load(audio_file)  # 替换为你自己的音乐文件路径
pygame.mixer.music.play(-1)  # -1表示循环播放


def check_end_map():
    if map_width+map_pos[0]<=win_width or map_height+map_pos[1]<=win_height:
        return True
    if map_pos[0]>0 or map_pos[1]>0:
        return True
    return False

# 游戏循环
while True:
    # 处理游戏事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # 移动小球
    ball_pos[0] += ball_speed[0]
    ball_pos[1] += ball_speed[1]

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        ball_pos[1] -= 5
    if keys[pygame.K_s]:
        ball_pos[1] += 5
    if keys[pygame.K_a]:
        ball_pos[0] -= 5
    if keys[pygame.K_d]:
        ball_pos[0] += 5


    # 更新地图位置

    if ball_pos[0] < win_width / 4:
        if not map_pos[0] > 0:
            map_pos[0] += ball_speed[0]
            ball_pos[0] += ball_speed[0]
    elif ball_pos[0] > win_width * 3 / 4:
        if not map_width+map_pos[0]<win_width:
            map_pos[0] -= ball_speed[0]
            ball_pos[0] -= ball_speed[0]
    if ball_pos[1] < win_height / 4:
        if not map_pos[1]>0:
            map_pos[1] += ball_speed[1]
            ball_pos[1] += ball_speed[1]
    elif ball_pos[1] > win_height * 3 / 4:
        if not  map_height+map_pos[1]<win_height:
            map_pos[1] -= ball_speed[1]
            ball_pos[1] -= ball_speed[1]

    # 检测小球是否碰到了地图边界
    if ball_pos[0] - ball_radius < map_pos[0] or ball_pos[0] + ball_radius > map_pos[0] + map_width:
        ball_speed[0] = -ball_speed[0]

    if ball_pos[1] - ball_radius < map_pos[1] or ball_pos[1] + ball_radius > map_pos[1] + map_height:
        ball_speed[1] = -ball_speed[1]

    # 帧计数器递增
    frame_count += 1

    # 如果当前帧数在目标帧数列表中，则随机改变小球颜色
    if frame_count in beat_frame:
        ball_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # ball_speed[0] = random.randint(-10, 10)
        # ball_speed[1] = random.randint(-10, 10)

    # 绘制背景和小球
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, map_color, (map_pos[0], map_pos[1], map_width, map_height))
    pygame.draw.circle(screen, ball_color, ball_pos, ball_radius)

    font = pygame.font.SysFont('Arial', 20)
    map_text = font.render('Map: ({}, {})'.format(map_pos[0], map_pos[1]), True, (0, 0, 0))
    task_text = font.render('Ball: ({}, {})'.format(ball_pos[0], ball_pos[1]), True, (0, 0, 0))
    screen.blit(map_text, (10, 10))
    screen.blit(task_text, (10, 30))
    # 更新屏幕
    pygame.display.update()