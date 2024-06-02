CREATE TABLE tmp
(
    id           bigint        NOT NULL AUTO_INCREMENT COMMENT '主键',
    run_id       varchar(64)   NOT NULL DEFAULT '' COMMENT '运行id',
    batch_run_id varchar(64)   NOT NULL DEFAULT '' COMMENT '批量运行id',
    seed         int(10)       NOT NULL DEFAULT 0 COMMENT '随机种子',
    create_time  timestamp     NOT NULL DEFAULT current_timestamp COMMENT '创建时间',
    PRIMARY KEY (id)
) ENGINE=innoDB DEFAULT CHARSET=utf8 comment '运行时配置';
